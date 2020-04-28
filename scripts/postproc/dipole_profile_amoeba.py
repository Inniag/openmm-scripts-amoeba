#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculates dipole profiles from existing AMOEBA trajectories.

The script will parse a parameter log file form the AMOEBA simulation script
and set up a simulation and OpenMM context similar to the one used in the
simulation. It will then go through an existing trajectory and for each set
of coordinates (i.e. for each frame) will calculate the AMOEBA atomic dipole
moments.

Three molecular dipole moments are then derived from these, corresponding to
the contributions from (i) permanent atomic dipoles, (ii) induced atomic
dipoles, and (iii) atomic charges/monopoles. Their (vector) sum is then con-
sidered to be the total molecular dipole moment. Molecular dipole moments are
converted to spherical coordinates, i.e. a set of dipole moment strength,
cosine of polar angle, and cosine of azimuthal angle.

All three components and the total dipole moment are calculated for each water
molecule within a cylinderical region around the protein's center of geometry.
The user needs to specify the radius of this cylinder to match the pore
dimensions and a margin for the extension of the cylinder beyond the bounding
box of the protein. This is done to save memory/disk space by ignoring
irrelevant water molecules.

The script also loads a CHAP detailed output file specifying a spatial curve
corresponding to the protein pore center line. This is used to convert the
water molecule z-coordinate to center line s-coordinate. This is done with
a static rather than per-frame curve for efficiency.

A JSON file containing a table of dipole profiles is writtin, where one
column specifies a number of bins.
"""


import argparse
import json
import sys

import numpy as np
import pandas as pd
from simtk.openmm import (AmoebaGeneralizedKirkwoodForce, AmoebaMultipoleForce,
                          AmoebaVdwForce, AmoebaWcaDispersionForce,
                          AndersenThermostat, CustomExternalForce,
                          MonteCarloAnisotropicBarostat, MTSIntegrator,
                          Platform, VerletIntegrator)
from simtk.openmm.app import PME, ForceField, PDBFile, Simulation
from simtk.unit import (Quantity, angstrom, bar, elementary_charge,
                        femtoseconds, kelvin, kilojoule_per_mole, nanometer,
                        picosecond)

import MDAnalysis as mda
from b_spline_curve import BSplineCurve


def sem(x):
    """
    Calculates standard error of the mean using numpy.
    """
    return np.std(x)/np.sqrt(np.size(x))


def qlo(x):
    """
    Calculates 25% quantile, lower quartile.
    """
    return np.percentile(x, 25)


def qhi(x):
    """
    Calculates 75% quantile, higher quartile.
    """
    return np.percentile(x, 100)


def cartesian2spherical(arr):
    """
    Expects a Nx3 array of Cartesian coordinates and returns an Nx3 array of
    spherical coordinates (i.e. rho, cos_theta, and cos_phi).
    """

    rho = np.linalg.norm(arr, axis=1)
    cos_theta = arr[:, 2] / rho
    cos_phi = np.cos(np.arctan2(arr[:, 1], arr[:, 0]))

    return(np.column_stack((rho, cos_theta, cos_phi)))


def pdb_file_nonstandard_bonds(filename):
    """Wraps the OpenMM PDBFile reader and adds bonds to nonstandard residues.

    This is necessary as the OpenMM mechanism of reading bonds from CONECT
    records in the PDB file fails if there are more than 9999 residues, which
    can easily happen for a bilayer system.
    """

    # load PDB file via OpenMM (this has missing bonds):
    pdb = PDBFile(filename)

    # load PDB file via MDA:
    u = mda.Universe(filename)

    # split into overall system and lipids:
    system = u.select_atoms("all")

    # select all nonstandard atoms:
    # (to avoid wasting time on bonds that OpenMM already knows)
    nonstandard = u.select_atoms(
        "not resname SOL and not resname HOH and not protein"
    )
    nonstandard_selection = "not protein"
    nonstandard_selection += " and not resname SOL"
    nonstandard_selection += " and not resname HOH"
    nonstandard_selection += " and not resname NA"
    nonstandard_selection += " and not resname CL"
    nonstandard = u.select_atoms(nonstandard_selection)

    # guess bonds in each individual residues:
    # (this prevents accidentally creating bonds between residues)
    print("--> guessing bonds for nonstandard residues")
    nonstandard_residues = [system.select_atoms(
        "resid " + str(resid)) for resid in list(set(nonstandard.resids))]
    for res in nonstandard_residues:
        res.guess_bonds()

    # add nonstandard bonds to OpenMM topology:
    atoms = (list(pdb.topology.atoms()))
    for bndidx in nonstandard.bonds.indices:
        pdb.topology.addBond(atoms[bndidx[0]], atoms[bndidx[1]])

    # return OpenMM PDB object:
    return pdb


def main(argdict):
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # load configuration from logfile:
    with open(argdict["log"], 'r') as f:
        argdict = json.load(f)

    # load system initial configuration:
    pdb = pdb_file_nonstandard_bonds(argdict["pdb"])
    print("--> input topology: ", end="")
    print(pdb.topology)

    # physical parameters of simulation:
    sim_temperature = argdict["temperature"] * kelvin
    sim_andersen_coupling = 1/picosecond
    sim_pressure = (
        (argdict["pressure"], argdict["pressure"], argdict["pressure"])*bar
    )
    sim_scale_x = True
    sim_scale_y = True
    sim_scale_z = True

    # simulation control parameters:
    sim_timestep = argdict["timestep"]*femtoseconds

    # restraints parameters:
    sim_restr_fc = argdict["restr_fc"]*kilojoule_per_mole/nanometer**2

    # create force field object:
    ff = ForceField(*argdict["ff"])

    # build a simulation system from topology and force field:
    # (note that AMOEBA is intended to be run without constraints)
    # (note that mutualInducedtargetEpsilon defaults to 0.01 unlike what is
    # specified in the documentation which claims 0.00001)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=argdict["nonbonded_cutoff"]*nanometer,
        vdwCutoff=argdict["vdw_cutoff"]*nanometer,
        ewaldErrorTolerance=argdict["ewald_error_tolerance"],
        polarisation=argdict["polarisation"],
        mutualInducedTargetEpsilon=argdict["mutual_induced_target_epsilon"],
        constraints=None,
        rigidWater=False,
        removeCMMotion=True     # removes centre of mass motion
    )

    # overwrite the polarisation method set at system creation; this is
    # necessary as openMM always sets polarisation method to "mutual" of the
    # target epsilon is specified at system creation; this way, target epsilon
    # is ignored for all but the mutual method
    multipole_force = [
        f for f in system.getForces() if isinstance(f, AmoebaMultipoleForce)
    ][0]
    print("--> using polarisation method " + str(argdict["polarisation"]))
    if argdict["polarisation"] == "mutual":
        multipole_force.setPolarizationType(multipole_force.Mutual)
    if argdict["polarisation"] == "extrapolated":
        multipole_force.setPolarizationType(multipole_force.Extrapolated)
    if argdict["polarisation"] == "direct":
        multipole_force.setPolarizationType(multipole_force.Direct)

    # will use Andersen thermostat here:
    # (Inhibits particle dynamics somewhat, but little or no ergodicity
    # issues (from Gromacs documenation). However, only alternative is full
    #  Langevin dynamics, which is even worse wrt dynamics. Bussi/v-rescale is
    # not available at the moment, it seems (it is available in tinker, but
    # without GPU acceleration))
    system.addForce(AndersenThermostat(
        sim_temperature,
        sim_andersen_coupling))

    # use anisotropic barostat:
    # (note that this corresponds to semiisotropic pressure coupling in Gromacs
    # if the pressure is identical for the x- and y/axes)
    # (note that by default this attempts an update every 25 steps)
    system.addForce(MonteCarloAnisotropicBarostat(
        sim_pressure,
        sim_temperature,
        sim_scale_x,
        sim_scale_y,
        sim_scale_z))

    # prepare harmonic restraining potential:
    # (note that periodic distance is absolutely necessary here to prevent
    # system from blowing up, as otherwise periodic image position may be used
    # resulting in arbitrarily large forces)
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", sim_restr_fc)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    # apply harmonic restraints to C-alphas:
    if argdict["restr"] == "capr":
        print("--> applying harmonic positional restraints to CA atoms")
        for atm in pdb.topology.atoms():
            if atm.name == "CA":
                force.addParticle(atm.index, pdb.positions[atm.index])
    elif argdict["restr"] == "hapr":
        sys.exit(
            "Restraints mode " + str(argdict["restr"]) + "is not implemented."
        )
    elif argdict["restr"] == "none":
        print("--> applying no harmonic positional restraints to any atom")
    else:
        sys.exit(
            "Restraints mode " + str(argdict["restr"]) + "is not implemented."
        )

    # add restraining force to system:
    system.addForce(force)

    # make special group for nonbonded forces:
    for f in system.getForces():
        if (
            isinstance(f, AmoebaMultipoleForce)
            or isinstance(f, AmoebaVdwForce)
            or isinstance(f, AmoebaGeneralizedKirkwoodForce)
            or isinstance(f, AmoebaWcaDispersionForce)
        ):
            f.setForceGroup(1)

    # select integrator:
    if argdict["integrator"] == "mts":
        # use multiple timestep RESPA integrator:
        print("--> using RESPA/MTS integrator")
        integrator = MTSIntegrator(
            sim_timestep, [(0, argdict["inner_ts_frac"]), (1, 1)]
        )
    if argdict["integrator"] == "verlet":
        # use Leapfrog Verlet integrator here:
        print("--> using Verlet integrator")
        integrator = VerletIntegrator(sim_timestep)

    # select a platform (should be CUDA, otherwise VERY slow):
    platform = Platform.getPlatformByName(argdict["platform"])
    properties = {"CudaPrecision": argdict["precision"], "CudaDeviceIndex": "0"}

    # create simulation system:
    sim = Simulation(pdb.topology, system, integrator, platform, properties)

    # unit conversion factors:
    ang2nm = 0.1

    # create MDA universe:
    u = mda.Universe(args.s, args.f)

    # selection for overall system will be needed to set OpenMM positions
    # accordingt to trajectory:
    allsystem = u.select_atoms("all")

    # get parameters to define cylinder around protein center of geometry:
    # (the cylinder spans the entire box in the z-direction)
    protein = u.select_atoms("protein")
    radius = str(args.r)
    z_margin = args.z_margin
    z_min = str(
        protein.bbox()[0, 2] - protein.center_of_geometry()[2] - z_margin
    )
    z_max = str(
        protein.bbox()[1, 2] - protein.center_of_geometry()[2] + z_margin
    )

    # select all solvent atoms, note that AMOEBA residue name is HOH:
    # (these must be updating, as water may move in and out of pore!)
    solvent = u.select_atoms(
        "byres (resname HOH SOL) and cyzone "
        + radius + " "
        + z_max + " "
        + z_min + " protein",
        updating=True
    )
    solvent_ow = solvent.select_atoms("name O OW", updating=True)

    # lambda function for converting atomic dipoles to molecular dipoles:
    # (this only works on 1D arrays, hence use apply_along_axis if quantity is
    # vector-valued, e.g. positions and dipoles)
    def atomic2molecular_sum(arr): return np.bincount(allsystem.resindices, arr)

    def atomic2molecular_avg(arr): return np.bincount(
        allsystem.resindices, arr) / np.bincount(allsystem.resindices)

    # create lambda function for obtaining charges in vectorisable way:
    # (units are elementary_charge)
    get_atomic_charges = np.vectorize(
        lambda index: multipole_force.getMultipoleParameters(int(index))[0]
        .value_in_unit(elementary_charge)
    )

    # obtain atomic charges:
    # (charges are static, so need this only once; units are elementary charge)
    atomic_charges = get_atomic_charges(allsystem.ix)

    # obtain start and end time as will as time step:
    dt = float(args.dt)
    t_start = float(args.b)
    t_end = float(args.e)

    # prepare results dictionary:
    res = {
        "t": [],
        "x": [],
        "y": [],
        "z": [],
        "indu_rho": [],
        "indu_costheta": [],
        "indu_cosphi": [],
        "perm_rho": [],
        "perm_costheta": [],
        "perm_cosphi": [],
        "mono_rho": [],
        "mono_costheta": [],
        "mono_cosphi": [],
        "total_rho": [],
        "total_costheta": [],
        "total_cosphi": []
    }

    # loop over trajectory:
    for ts in u.trajectory:

        # skip all frames before starting frame:
        if ts.time < t_start:
            continue

        # only analyse relevant time frames:
        if round(ts.time, 4) % dt == 0:

            # inform user:
            print(
                "analysing frame: "
                + str(ts.frame)
                + " at time: "
                + str(ts.time)
            )
            print(
                "number of selected solvent molecules in this frame: "
                + str(solvent.n_residues)
            )

            # convert mda positions to OpenMM positions and set context:
            omm_positions = Quantity(
                [tuple(pos) for pos in list(allsystem.positions)],
                unit=angstrom
            )
            sim.context.setPositions(omm_positions)

            # calculate molecular positions (or molecular centre of geometry) by
            # averaging over all atomic positions within a residue:
            # (units are Angstrom in MDAnalysis!)
            molecular_positions = np.apply_along_axis(
                atomic2molecular_avg, 0, allsystem.positions) * ang2nm

            # calculate charge-weighted positions by multiplying the relative
            # atomic positions with the atomic charges (relative positions are
            # necessary to account for charged residues/molecules, where the
            # dipole moment is calculated relative to the center of geometry of
            # the residue):
            # (units are elementary charge * nanometer)
            atomic_charge_weighted_positions = (
                allsystem.positions - molecular_positions[allsystem.resindices]
            )
            atomic_charge_weighted_positions *= (
                atomic_charges[np.newaxis].T * ang2nm
            )

            # obtain induced and permanent atomic dipoles from OpenMM:
            # (units are elementary charge * nm)
            atomic_dipoles_indu = np.array(
                multipole_force.getInducedDipoles(sim.context)
            )
            atomic_dipoles_perm = np.array(
                multipole_force.getLabFramePermanentDipoles(sim.context)
            )

            # convert atomic to molecular quantities and calculate total dipole:
            molecular_dipoles_indu = np.apply_along_axis(
                atomic2molecular_sum, 0, atomic_dipoles_indu)
            molecular_dipoles_perm = np.apply_along_axis(
                atomic2molecular_sum, 0, atomic_dipoles_perm)
            molecular_dipoles_mono = np.apply_along_axis(
                atomic2molecular_sum, 0, atomic_charge_weighted_positions)
            molecular_dipoles_total = (
                molecular_dipoles_indu
                + molecular_dipoles_perm
                + molecular_dipoles_mono
            )

            # convert to spherical coordinates:
            molecular_dipoles_indu = cartesian2spherical(molecular_dipoles_indu)
            molecular_dipoles_perm = cartesian2spherical(molecular_dipoles_perm)
            molecular_dipoles_mono = cartesian2spherical(molecular_dipoles_mono)
            molecular_dipoles_total = cartesian2spherical(
                molecular_dipoles_total
            )

            # insert into results dictionary:
            res["t"].append(np.repeat(ts.time, solvent.n_residues))
            res["x"].append(molecular_positions[solvent_ow.resindices, 0])
            res["y"].append(molecular_positions[solvent_ow.resindices, 1])
            res["z"].append(molecular_positions[solvent_ow.resindices, 2])
            res["indu_rho"].append(
                molecular_dipoles_indu[solvent_ow.resindices, 0]
            )
            res["indu_costheta"].append(
                molecular_dipoles_indu[solvent_ow.resindices, 1]
            )
            res["indu_cosphi"].append(
                molecular_dipoles_indu[solvent_ow.resindices, 2]
            )
            res["perm_rho"].append(
                molecular_dipoles_perm[solvent_ow.resindices, 0]
            )
            res["perm_costheta"].append(
                molecular_dipoles_perm[solvent_ow.resindices, 1]
            )
            res["perm_cosphi"].append(
                molecular_dipoles_perm[solvent_ow.resindices, 2]
            )
            res["mono_rho"].append(
                molecular_dipoles_mono[solvent_ow.resindices, 0]
            )
            res["mono_costheta"].append(
                molecular_dipoles_mono[solvent_ow.resindices, 1]
            )
            res["mono_cosphi"].append(
                molecular_dipoles_mono[solvent_ow.resindices, 2]
            )
            res["total_rho"].append(
                molecular_dipoles_total[solvent_ow.resindices, 0]
            )
            res["total_costheta"].append(
                molecular_dipoles_total[solvent_ow.resindices, 1]
            )
            res["total_cosphi"].append(
                molecular_dipoles_total[solvent_ow.resindices, 2]
            )

        # stop iterating through trajectory after end time:
        if ts.time > t_end:
            break

    # convert lists of arrays to arrays:
    for k in res.keys():
        res[k] = np.concatenate(res[k])

    # convert units of dipole magnitude to Debye:
    eNm2debye = 48.03205
    res["indu_rho"] = eNm2debye*res["indu_rho"]
    res["perm_rho"] = eNm2debye*res["perm_rho"]
    res["mono_rho"] = eNm2debye*res["mono_rho"]
    res["total_rho"] = eNm2debye*res["total_rho"]

    # load spline curve data:
    with open(args.j, "r") as f:
        chap_data = json.load(f)

    # create spline curve from CHAP data:
    spline_curve = BSplineCurve(chap_data)

    # calculate s-coordinate from z-coordinate:
    res["s"] = spline_curve.z2s(res["z"])

    # convert results to data frame:
    df_res = pd.DataFrame(res)

    # loop over various numbers of bins:
    df = []
    for nbins in args.nbins:

        # create a temporary data frame:
        tmp = df_res

        # drop positional coordinates:
        tmp = tmp.drop(["x", "y", "z", "t"], axis=1)

        # bin by value of s-coordinate:
        tmp = tmp.groupby(pd.cut(tmp.s, nbins))

        # aggregate variables:
        tmp = tmp.agg(
            [np.mean, np.std, sem, np.size, np.median, qlo, qhi]
        ).reset_index()

        # rename columns (combines variable name with aggregation method):
        tmp.columns = ["_".join(x) for x in tmp.columns.ravel()]

        # remove grouping key:
        tmp = tmp.drop("s_", axis=1)

        # add column wit number of bins:
        tmp["nbins"] = nbins

        # append to list of data frames:
        df.append(tmp)

    # combine list of data frames into single data frame:
    df = pd.concat(df)

    # write to JSON file:
    df.to_json(args.o, orient="records")

    # need to add newline for POSIX compliance:
    with open(args.o, "a") as f:
        f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-log",
        type=str,
        default="production_parameters.log",
        help="Name of logfile from original simulation."
    )
    parser.add_argument(
        "-j",
        nargs=None,
        default="stream_output.json",
        help="Name CHAP detailed output file."
    )
    parser.add_argument(
        "-s",
        type=str,
        default="atomistic_system_openmm.pdb",
        help="Name of structure file to be used as MDA topology."
    )
    parser.add_argument(
        "-f",
        type=str,
        default="production.xtc",
        help="Name of trajectory file to post-process."
    )
    parser.add_argument(
        "-o",
        nargs=1,
        default="dipole_moment.json",
        help="Name of output JSON file."
    )
    parser.add_argument(
        "-dt",
        type=float,
        default=1000,
        help="Post-processing time step in ps."
    )
    parser.add_argument(
        "-b",
        type=float,
        default=0.0,
        help="Starting time for post-processing in ps."
    )
    parser.add_argument(
        "-e",
        default=float("Inf"),
        help="End time for post-processing in ps."
    )
    parser.add_argument(
        "-nbins",
        type=float,
        nargs="+",
        default=[10, 50, 100],
        help="Array of number of bins to use in profile aggregation."
    )
    parser.add_argument(
        "-r",
        default=10.0,
        help="Radius of cylindrical selection for water molecules in Angstrom."
    )
    parser.add_argument(
        "-z_margin",
        default=10.0,
        type=float,
        help="Extrapolation beyond protein bounding box in z-direction for"
        "cylindrical selection in Angstrom."
    )

    args = parser.parse_args()
    argdict = vars(args)

    main(argdict)
