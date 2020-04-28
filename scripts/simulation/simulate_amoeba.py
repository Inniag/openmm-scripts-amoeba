#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Performs MD simulation using the AMOEBA polarisable force field.

Performs equilibrium MD simulation in the NPT ensemble where the alpha carbons
of the simulated protein are subjected to harmonic positional restraints. Prior
to productive sampling, a user defined number of energy minimisation steps will
be carried out. If the input system has already been equilibrated, not many
steps will be needed. Minimisation will not be repeated if the script is
started from a log-file as input (in this case, the simulation will simply be
continued from the checkpoint file specified in the log-file).
"""

import argparse
import json
from sys import stdout

from simtk.openmm import (AmoebaGeneralizedKirkwoodForce, AmoebaMultipoleForce,
                          AmoebaVdwForce, AmoebaWcaDispersionForce,
                          AndersenThermostat, CustomExternalForce,
                          MonteCarloAnisotropicBarostat, MTSIntegrator,
                          Platform, VerletIntegrator)
from simtk.openmm.app import (PME, CheckpointReporter, DCDReporter, ForceField,
                              PDBFile, Simulation, StateDataReporter)
from simtk.unit import (bar, femtoseconds, kelvin, kilojoule_per_mole,
                        nanometer, picosecond)

import MDAnalysis as mda


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
    nonstandard_residues = [
        system.select_atoms("resid " + str(resid))
        for resid in list(set(nonstandard.resids))
    ]

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

    # are we continuing from logfile or starting from fresh PDB?
    if args.log is None:

        # keep track of restart number:
        args.restart_number = int(0)

        # write arguments to a file to keep a record:
        with open(args.outname + "_parameters.log", 'w') as f:
            print(json.dumps(vars(args), sort_keys=True, indent=4), file=f)
    else:

        # load configuration from logfile:
        with open(args.log, 'r') as f:
            argdict = json.load(f)

            # increment restart number:
        argdict["restart_number"] += 1

        with open(argdict["outname"] + "_parameters.log", 'w') as f:
            print(json.dumps(argdict, sort_keys=True, indent=4), file=f)

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
    sim_num_steps = argdict["num_steps"]
    sim_traj_rep_steps = argdict["report_freq"]
    sim_state_rep_steps = argdict["report_freq"]
    sim_checkpoint_steps = argdict["report_freq"]
    sim_energy_minimisation_tolerance = (
        argdict["minimisation_tolerance"] * kilojoule_per_mole
    )
    sim_energy_minimisation_steps = argdict["minimisation_steps"]

    # restraints parameters:
    sim_restr_fc = argdict["restr_fc"]*kilojoule_per_mole/nanometer**2

    # create force field object:
    ff = ForceField(*argdict["ff"])

    # build a simulation system from topology and force field:
    # (note that AMOEBA is intended to be run without constraints)
    # (note that mutualInducedtargetEpsilon defaults to 0.01 unlike what is
    # specified in the documentation which claims 0.00001)
    sys = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=argdict["nonbonded_cutoff"]*nanometer,
        vdwCutoff=argdict["vdw_cutoff"]*nanometer,
        ewaldErrorTolerance=argdict["ewald_error_tolerance"],
        polarisation=argdict["polarisation"],
        mutualInducedTargetEpsilon=argdict["mutual_induced_target_epsilon"],
        constraints=None,
        rigidWater=False,
        removeCMMotion=True    # removes centre of mass motion
    )

    # overwrite the polarisation method set at system creation; this is
    # necessary as openMM always sets polarisation method to "mutual" of the
    # target epsilon is specified at system creation; this way, target epsilon
    # is ignored for all but the mutual method
    multipole_force = sys.getForce(9)
    print("--> using polarisation method " + str(argdict["polarisation"]))
    if args.polarisation == "mutual":
        multipole_force.setPolarizationType(multipole_force.Mutual)
    if args.polarisation == "extrapolated":
        multipole_force.setPolarizationType(multipole_force.Extrapolated)
    if args.polarisation == "direct":
        multipole_force.setPolarizationType(multipole_force.Direct)

    # will use Andersen thermostat here:
    # (Inhibits particle dynamics somewhat, but little or no ergodicity
    # issues (from Gromacs documenation). However, only alternative is full
    # Langevin dynamics, which is even worse wrt dynamics. Bussi/v-rescale is
    # not available at the moment, it seems (it is available in tinker, but
    # without GPU acceleration))
    sys.addForce(AndersenThermostat(
        sim_temperature,
        sim_andersen_coupling))

    # use anisotropic barostat:
    # (note that this corresponds to semiisotropic pressure coupling in Gromacs
    # if the pressure is identical for the x- and y/axes)
    # (note that by default this attempts an update every 25 steps)
    sys.addForce(MonteCarloAnisotropicBarostat(
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
    sys.addForce(force)

    # make special group for nonbonded forces:
    for f in sys.getForces():
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
            sim_timestep, [(0, args.inner_ts_frac), (1, 1)]
        )

    if argdict["integrator"] == "verlet":
        # use Leapfrog Verlet integrator here:
        print("--> using Verlet integrator")
        integrator = VerletIntegrator(sim_timestep)

    # select a platform:
    platform = Platform.getPlatformByName(argdict["platform"])

    # additional settings for simulation on GPU:
    properties = {}
    if argdict["platform"] == "CUDA":
        properties = {
            "CudaPrecision": argdict["precision"], "CudaDeviceIndex": "0"
        }

    # prepare a simulation from topology, system, and integrator and set initial
    # positions as in PDB file:
    sim = Simulation(pdb.topology, sys, integrator, platform, properties)
    sim.context.setPositions(pdb.positions)

    if args.log is None:

        # perform energy minimisation:
        print("--> performing energy minimsation")
        sim.minimizeEnergy(
            tolerance=sim_energy_minimisation_tolerance,
            maxIterations=sim_energy_minimisation_steps)

        # set starting velocities:
        print("--> generating random starting velocities")
        sim.context.setVelocitiesToTemperature(argdict["temperature"] * kelvin)

    else:

        # load checkpoint file:
        sim.loadCheckpoint(argdict["outname"] + ".chk")

    # write simulation trajectory to DCD file:
    dcd_outname = (
        argdict["outname"] + "_" + str(argdict["restart_number"]) + str(".dcd")
    )
    sim.reporters.append(DCDReporter(dcd_outname, sim_traj_rep_steps))

    # write state data to tab-separated CSV file:
    state_outname = (
        argdict["outname"] + "_" + str(argdict["restart_number"]) + str(".csv")
    )
    sim.reporters.append(StateDataReporter(
        state_outname,
        sim_state_rep_steps,
        step=True,
        time=True,
        progress=False,
        remainingTime=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=False,
        speed=True,
        totalSteps=sim_num_steps,
        separator="\t"))

    # write limited state information to standard out:
    sim.reporters.append(StateDataReporter(
        stdout,
        sim_state_rep_steps,
        step=True,
        time=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=sim_num_steps,
        separator="\t"))

    # write checkpoint files regularly:
    sim.reporters.append(CheckpointReporter(
        argdict["outname"] + ".chk",
        sim_checkpoint_steps))

    # advance simulation:
    print("--> beginning integration")
    sim.step(sim_num_steps)

    # save final simulation state:
    sim.saveState(argdict["outname"] + ".xml")
    positions = sim.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(
        sim.topology, positions, open(argdict["outname"] + ".pdb", "w")
    )


# entry point check:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-log",
        nargs="?",
        const="system_openmm.pdb",
        help="Name of logfile from which to continue simulation.")
    group.add_argument(
        "-pdb",
        nargs="?",
        const="system_openmm.pdb",
        default="system_openmm.pdb",
        help="Input structure PDB file.")

    parser.add_argument(
        "-ff",
        nargs="+",
        default="amoeba2013.xml",
        help="List of force field XML files. Note that default amoeba2013.xml "
        "has AMOEBA03 water model.")
    parser.add_argument(
        "-outname",
        nargs="?",
        const="openmm",
        default="openmm",
        help="Name prefix for all output files.")
    parser.add_argument(
        "-platform",
        nargs="?",
        const="CUDA",
        default="CUDA",
        help="Platform on which to simulation. Can be CUDA, CPU, or Reference.")
    parser.add_argument(
        "-precision",
        nargs="?",
        const="mixed",
        default="mixed",
        help="Floating point precision mode. Mixed assembles forces in single "
        "precision, but integrates in double precision.")
    parser.add_argument(
        "-minimisation_tolerance",
        nargs="?",
        type=float,
        const=1.0,
        help="Tolerance threshold for energy minimiser in kilojoule/mole.",
        default=1.0)
    parser.add_argument(
        "-minimisation_steps",
        nargs="?",
        type=int,
        const=1000,
        help="Maximum number of energy minimsation steps.",
        default=1000)
    parser.add_argument(
        "-polarisation",
        nargs="?",
        const="extrapolated",
        default="extrapolated",
        help="Method for calculating induced dipoles. Can be mutual,"
        " extrapolated, or direct.")
    parser.add_argument(
        "-mutual_induced_target_epsilon",
        type=float,
        nargs="?",
        const=1e-5,
        default=1e-5,
        help="Relative error tolerance for coverging induced dipoles in"
        " self-consistent field (mutual) approach.")
    parser.add_argument(
        "-nonbonded_cutoff",
        type=float,
        nargs="?",
        const=0.8,		# value used in AMOEBA parameterisation
        default=0.8,
        help="Cutoff for all non-bonded forces except van der Waals in nm.")
    parser.add_argument(
        "-vdw_cutoff",
        type=float,
        nargs="?",
        const=1.2,		# value used in AMOEBA parameterisation
        default=1.2,
        help="Cutoff for van der Waals forces in nm.")
    parser.add_argument(
        "-ewald_error_tolerance",
        type=float,
        nargs="?",
        const=5e-4,		# the default in openMM
        default=5e-4,
        help="Relative tolerance for PME calculations.")
    parser.add_argument(
        "-integrator",
        nargs="?",
        const="mts",
        default="mts",
        help="Time stepping algorithm. Can be verlet or mts (RESPA).")
    parser.add_argument(
        "-timestep",
        type=float,
        nargs="?",
        const=3.5,
        default=3.5,
        help="Time step in femtoseconds. For MTS integrator this is the outer"
        " time step.")
    parser.add_argument(
        "-inner_ts_frac",
        type=int,
        nargs="?",
        const=8,
        default=8,
        help="Inner time step as fraction of out time step. Ignored for Verlet "
        " integrator.")
    parser.add_argument(
        "-num_steps",
        type=int,
        nargs="?",
        const=100,
        default=100,
        help="Number of time steps to compute.")
    parser.add_argument(
        "-report_freq",
        type=int,
        nargs="?",
        const=25,
        default=25,
        help="Write output every report_freq steps.")
    parser.add_argument(
        "-temperature",
        type=float,
        nargs="?",
        const=310,
        default=310,
        help="Temperature in Kelvin.")
    parser.add_argument(
        "-pressure",
        type=float,
        nargs="?",
        const=1,
        default=1,
        help="Pressure in bar.")
    parser.add_argument(
        "-restr",
        type=str,
        nargs="?",
        const="capr",
        default="capr",
        help="Group of atoms to apply harmonic positional restraints to. Can"
        " be capr, hapr, or none.")
    parser.add_argument(
        "-restr_fc",
        type=float,
        nargs="?",
        const=1000,
        default=1000,
        help="Force constant for harmonic positional restraints in"
        " kilojoule/mol/nm^2.")
    args = parser.parse_args()
    argdict = vars(args)

    # pass arguments to main function:
    main(argdict)
