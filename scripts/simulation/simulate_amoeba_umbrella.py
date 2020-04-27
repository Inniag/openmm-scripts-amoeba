#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Performs MD simulation in an individual umbrella window using the AMOEBA FF.

Performs equilibrium MD simulation in the NPT ensemble where the alpha carbons
of the simulated protein are subjected to harmonic position restraints. In
addition, the z-coordinate of an umbrella target particle is also subjected to
harmonic restraints (i.e. a biasing potential). Specification of which particle
to restrain as well as force constant and collective variable value are read
from a user-specified JSON file.

This JSON file needs to specify the index of the atom to be restrained, the
value of the collective variable (z-coordinate) at which to restrain this
particle, the coordinates at which to place the target particle initially, and
the force constant for the biasing potential. The particle with the specified
index will be moved to the position specified as CV in the JSON file. Its
x/y-coordinates will also be read from the JSON file, but will only be used for
initial placement and will not be subject to a biasing potential.

Prior to productive sampling, a user defined number of energy minimisation steps
will be carried out. If the input system has already been equilibrated, not
many steps will be needed. Following minimisation, a user-specified number of
equilibration steps will be carried out with a ten-fold reduced time step. These
measures prevent clashes and usually lead to stable simulation. Minimisation and
equilibration will not be repeated if the script is started with a log-file as
input (in this case, the simulation will simply be continued from the checkpoint
file specified in the log-file).

The value of the collective variable over time is written to space-separated
text file, which can be used for further processing such as WHAM analysis.
"""

import argparse
import json
import os
import xml.etree.ElementTree as et
from sys import stdout

from simtk.openmm import (AmoebaGeneralizedKirkwoodForce, AmoebaMultipoleForce,
                          AmoebaVdwForce, AmoebaWcaDispersionForce,
                          AndersenThermostat, CustomExternalForce,
                          MonteCarloAnisotropicBarostat, MTSIntegrator,
                          Platform, VerletIntegrator)
from simtk.openmm.app import (PME, CheckpointReporter, DCDReporter, ForceField,
                              PDBFile, Simulation, StateDataReporter)
from simtk.unit import (Quantity, bar, femtoseconds, kelvin,
                        kilojoule_per_mole, nanometer, picosecond)

import MDAnalysis as mda

__author__ = "Gianni Klesse"
__copyright__ = "Copyright 2019, Gianni Klesse"
__credits__ = ["Gianni Klesse"]


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
        """not resname SOL and not resname HOH and not protein"""
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
    nonstandard_residues = (
        [
            system.select_atoms("resid " + str(resid))
            for resid in list(set(nonstandard.resids))
        ]
    )
    for res in nonstandard_residues:
        res.guess_bonds()

    # add nonstandard bonds to OpenMM topology:
    # WARNING: this will add bonds even if they are already present! better
    # delete all CONECT records from input PDBs!
    atoms = (list(pdb.topology.atoms()))
    for bndidx in nonstandard.bonds.indices:
        pdb.topology.addBond(atoms[bndidx[0]], atoms[bndidx[1]])

    # return OpenMM PDB object:
    return pdb


def atom_idx_from_bonds(topology, idx):
    """Finds indices of atoms bonded to a given atom.

    Simply loops over all bonds in topology and checks whether atom with given
    index is involved. Returns a list of indeces of all atoms bonded to the
    atom with given index.
    """

    # list of indeces of bonded atoms:
    idx_bonded = []

    # check all bonds:
    for bond in topology.bonds():

        if idx == bond.atom1.index:
            idx_bonded.append(bond.atom2.index)

        if idx == bond.atom2.index:
            idx_bonded.append(bond.atom1.index)

    # return indeces of bonded atoms:
    return(idx_bonded)


def periodic_box_vectors_from_xml(xmlfile):
    """Extracts periodic box vectors from OpenMM XML state file.

    Box vectors are returned in the format expected by the box vector setting
    function of OpenMM's topology.
    """

    # parse XML file:
    tree = et.parse(xmlfile)
    root = tree.getroot()

    # name of box vector field in XML file:
    pbv = "PeriodicBoxVectors"

    # box vectors need to be tuple of tuples:
    box_vectors = tuple([
        tuple([float(x) for x in root.find(pbv).find("A").attrib.values()]),
        tuple([float(x) for x in root.find(pbv).find("B").attrib.values()]),
        tuple([float(x) for x in root.find(pbv).find("C").attrib.values()])
    ])

    # add units:
    box_vectors = Quantity(box_vectors, nanometer)

    # return dimensions to caller:
    return(box_vectors)


class CollectiveVariableReporter(object):
    """Custom reporter for writing the value of the collective variable.

    Writes the z-coordinate of the umbrella target particle to a file.
    """

    def __init__(self, file, reportInterval, idx):
        """Constructs reporter.

        Sets index of particle that will be monitored by this reporter.
        """

        self._out = open(file, 'w')
        self._idx = idx
        self._reportInterval = reportInterval
        self._needsPositions = True
        self._needsVelocities = False
        self._needsForces = False
        self._needsEnergy = False

    def describeNextReport(self, simulation):
        """Returns a description of the next report generated by this reporter.

        Returns the number of steps till next report plus the requirements for
        various system properties the reporter needs.
        """

        # calculate number of steps till next report:
        steps = self._reportInterval
        steps -= simulation.currentStep % self._reportInterval

        # return report description:
        return (
            steps,
            self._needsPositions,
            self._needsVelocities,
            self._needsForces,
            self._needsEnergy
        )

    def report(self, simulation, state):
        """Generates a report.

        Writes output data to file.
        """

        # obtain report values:
        time = (
            state.getTime().value_in_unit(picosecond)
        )
        cv = (
            state.getPositions()[self._idx][2].value_in_unit(nanometer)
        )
        values = [time, cv]

        # print values:
        print(" ".join(str(v) for v in values), file=self._out)

        # flush write buffer:
        try:
            self._out.flush()
        except AttributeError:
            pass


def main(argdict):
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # are we continuing from logfile or starting from fresh PDB?
    if argdict["log"] is None:

        # keep track of restart number:
        argdict["restart_number"] = int(0)

        # write arguments to a file to keep a record:
        with open(argdict["outname"] + "_parameters.log", 'w') as f:
            print(json.dumps(argdict, sort_keys=True, indent=4), file=f)
    else:

        # load configuration from logfile:
        logfile = argdict["log"]
        with open(argdict["log"], 'r') as f:
            argdict = json.load(f)

        # make sure log file has appropriate entry:
        argdict["log"] = logfile

        # increment restart number:
        argdict["restart_number"] += 1

        # write unnumbered parameters file (for easy restarts):
        with open(argdict["outname"] + "_parameters.log", 'w') as f:
            print(json.dumps(argdict, sort_keys=True, indent=4), file=f)

    # write numbered parameters file (for record keeping):
    with open(
        argdict["outname"]
        + "_" + str(argdict["restart_number"])
        + "_parameters.log", 'w'
    ) as f:
        print(json.dumps(argdict, sort_keys=True, indent=4), file=f)

    # load system initial configuration:
    pdb = pdb_file_nonstandard_bonds(argdict["pdb"])
    print("--> input topology: ", end="")
    print(pdb.topology)

    # get XML file from absolute path:
    xmlfile = os.path.abspath(argdict["xml"])

    # set box size in topology to values from XML file:
    box_vectors = periodic_box_vectors_from_xml(xmlfile)
    pdb.topology.setPeriodicBoxVectors(box_vectors)

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
        removeCMMotion=True  # removes centre of mass motion
    )

    # overwrite the polarisation method set at system creation; this is
    # necessary as openMM always sets polarisation method to "mutual" of the
    # target epsilon is specified at system creation; this way, target epsilon
    # is ignored for all but the mutual method
    multipole_force = sys.getForce(9)
    print("--> using polarisation method " + str(argdict["polarisation"]))
    if argdict["polarisation"] == "mutual":
        multipole_force.setPolarizationType(multipole_force.Mutual)
    elif argdict["polarisation"] == "extrapolated":
        multipole_force.setPolarizationType(multipole_force.Extrapolated)
    elif argdict["polarisation"] == "direct":
        multipole_force.setPolarizationType(multipole_force.Direct)
    else:
        raise Exception(
            "Polarisation method "
            + str(argdict["polarisation"])
            + " not supported!"
        )

    # will use Andersen thermostat here:
    # (Inhibits particle dynamics somewhat, but little or no ergodicity
    # issues (from Gromacs documenation). However, only alternative is full
    # Langevin dynamics, which is even worse wrt dynamics. Bussi/v-rescale is
    # not available at the moment, it seems (it is available in tinker, but
    # without GPU acceleration))
    sys.addForce(
        AndersenThermostat(sim_temperature, sim_andersen_coupling)
    )

    # use anisotropic barostat:
    # (note that this corresponds to semiisotropic pressure coupling in Gromacs
    # if the pressure is identical for the x- and y/axes)
    # (note that by default this attempts an update every 25 steps)
    sys.addForce(
        MonteCarloAnisotropicBarostat(
            sim_pressure,
            sim_temperature,
            sim_scale_x,
            sim_scale_y,
            sim_scale_z
        )
    )

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

    # read umbrella parameters from file:

    with open(argdict["umbrella_target"], "r") as f:
        umbrella_target = json.load(f)

    # obtain index from atom to be restrained:
    umbrella_index = umbrella_target["target_particle"]["index"]
    umbrella_fc = (
        umbrella_target["umbrella_params"]["fc"]*kilojoule_per_mole/nanometer**2
    )
    umbrella_cv = umbrella_target["umbrella_params"]["cv"]*nanometer

    # inform user:
    print(
        "--> applying umbrella potential to "
        + str(list(pdb.topology.atoms())[umbrella_index])
        + " at position "
        + str(pdb.positions[umbrella_index])
    )

    # additional restraining force applied to ion under umbrella restraints:
    umbrella_force = CustomExternalForce(
        "k*periodicdistance(0.0, 0.0, z, 0.0, 0.0, z0)^2"
    )
    umbrella_force.addGlobalParameter("k", umbrella_fc)
    # z0 is set to value in JSON file rather than initial particle coordinate to
    # allow for a few steps of energy minimisation to avoid clashes between the
    # restrained umbrella target and surrounding atoms:
    umbrella_force.addGlobalParameter("z0", umbrella_cv)

    # select integrator:
    if argdict["integrator"] == "mts":

        # use multiple timestep RESPA integrator:
        print("--> using RESPA/MTS integrator")
        integrator = MTSIntegrator(
            sim_timestep,
            [(0, argdict["inner_ts_frac"]), (1, 1)]
        )

    elif argdict["integrator"] == "verlet":

        # use Leapfrog Verlet integrator here:
        print("--> using Verlet integrator")
        integrator = VerletIntegrator(sim_timestep)

    else:

        # no other integrators supported:
        raise Exception(
            "Integrator "
            + str(argdict["integrator"])
            + " is not supported."
        )

    # select a platform:
    platform = Platform.getPlatformByName(argdict["platform"])
    properties = {"CudaPrecision": argdict["precision"], "CudaDeviceIndex": "0"}

    # prepare a simulation from topology, system, and integrator and set initial
    # positions as in PDB file:
    sim = Simulation(pdb.topology, sys, integrator, platform, properties)

    # is this initial simulation or restart from checkpoint?
    if argdict["log"] is None:

        # load positions and velocities from XML file:
        print("--> loading simulation state from XML file...")
        sim.loadState(xmlfile)

        # find all particles bonded to ion (i.e. Drude particles):
        idx_bonded = atom_idx_from_bonds(
            sim.topology,
            umbrella_index
        )
        idx_shift = idx_bonded + [umbrella_index]

        # shift target particle into position:
        pos = (
            sim.context
            .getState(getPositions=True)
            .getPositions(asNumpy=True)
        )
        pos[idx_shift, :] = (
            umbrella_target["target_particle"]["init_pos"] * nanometer
        )
        print("--> target particle now placed at " + str(pos[idx_shift, :]))

        # set new particle positions in context:
        sim.context.setPositions(pos)
        e_pot = sim.context.getState(getEnergy=True).getPotentialEnergy()
        print("--> potential energy after target placement is: " + str(e_pot))

        # minimise energy to remove clashes:
        # (too many steps might ruin everythin!)
        print("--> running energy minimisation...")
        sim.minimizeEnergy(maxIterations=argdict["minimisation_steps"])
        e_pot = sim.context.getState(getEnergy=True).getPotentialEnergy()
        print(
            "--> "
            + str(argdict["minimisation_steps"])
            + " steps of energy minimisation reduced the potential energy to "
            + str(e_pot)
        )

        # reduce time step for equilibration period:
        print("--> running equilibration at reduced time step...")
        sim.integrator.setStepSize(
            0.1*sim.integrator.getStepSize()
        )
        sim.context.setTime(0.0 * picosecond)

        # will write report about equilibration phase:
        sim.reporters.append(
            StateDataReporter(
                stdout,
                int(argdict["equilibration_steps"] / 10),
                step=True,
                time=True,
                speed=True,
                progress=True,
                remainingTime=True,
                totalSteps=argdict["equilibration_steps"],
                separator="\t"
            )
        )

        # run equilibration steps:
        sim.step(argdict["equilibration_steps"])

        # reset step size to proper value:
        sim.integrator.setStepSize(
            10.0*sim.integrator.getStepSize()
        )
        sim.context.setTime(0.0 * picosecond)
        sim.reporters.clear()

    else:

        # load checkpoint file:
        checkpoint_file = (
            str(argdict["outname"])
            + "_"
            + str(argdict["restart_number"] - 1)
            + ".chk"
        )
        print("--> loading checkpoint file " + checkpoint_file)
        sim.loadCheckpoint(checkpoint_file)

    # write collective variable value to file:
    sample_outname = (
        argdict["outname"] + "_" + str(argdict["restart_number"]) + str(".dat")
    )
    sim.reporters.append(
        CollectiveVariableReporter(
            sample_outname,
            argdict["umbrella_freq"],
            umbrella_index
        )
    )

    # write simulation trajectory to DCD file:
    dcd_outname = (
        argdict["outname"] + "_" + str(argdict["restart_number"]) + str(".dcd")
    )
    sim.reporters.append(
        DCDReporter(
            dcd_outname,
            sim_traj_rep_steps
        )
    )

    # write state data to tab-separated CSV file:
    state_outname = (
        argdict["outname"] + "_" + str(argdict["restart_number"]) + str(".csv")
    )
    sim.reporters.append(
        StateDataReporter(
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
            separator="\t"
        )
    )

    # write limited state information to standard out:
    sim.reporters.append(
        StateDataReporter(
            stdout,
            sim_state_rep_steps,
            step=True,
            time=True,
            speed=True,
            progress=True,
            remainingTime=True,
            totalSteps=sim_num_steps,
            separator="\t"
        )
    )

    # write checkpoint files regularly:
    checkpoint_outname = (
        argdict["outname"] + "_" + str(argdict["restart_number"]) + ".chk"
    )
    sim.reporters.append(
        CheckpointReporter(
            checkpoint_outname,
            sim_checkpoint_steps
        )
    )

    # run simulation:
    sim.step(argdict["num_steps"])

    # save final simulation state:
    sim.saveState(
        argdict["outname"] + "_" + str(argdict["restart_number"]) + ".xml"
    )
    positions = sim.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(
        sim.topology,
        positions,
        open(
            argdict["outname"]
            + "_" + str(argdict["restart_number"])
            + ".pdb", "w"
        )
    )


# entry point check:
if __name__ == "__main__":

    # parse command line arguments:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-log",
        nargs="?",
        const="system_openmm.pdb",
        help="""Name of logfile from which to cintinue simulation."""
    )
    group.add_argument(
        "-pdb",
        nargs=None,
        default="umbrella_input.pdb",
        help="""Input structure PDB file. System topology will be determined
        from this file."""
    )
    parser.add_argument(
        "-xml",
        nargs=None,
        default="umbrella_input.xml",
        help="""OpenMM XML state file containing an equilibrated simulation
        system. Starting positions and box dimensions will be taken from this
        file."""
    )
    parser.add_argument(
        "-ff",
        nargs="+",
        default="amoeba2013.xml",
        help="""List of force field XML files. Note that default
        amoeba2013.xml has AMOEBA03 water model."""
    )
    parser.add_argument(
        "-umbrella_target",
        nargs=None,
        default="umbrella_target.json",
        help="""JSON file specifying which particle should be placed under
        umbrella restraints."""
    )
    parser.add_argument(
        "-umbrella_freq",
        type=int,
        nargs=None,
        default=25,
        help="""Write umbrella output every umbrella_freq steps."""
    )
    parser.add_argument(
        "-outname",
        nargs=None,
        type=str,
        default="umbrella",
        help="""Name prefix for all output files."""
    )
    parser.add_argument(
        "-platform",
        nargs="?",
        const="CUDA",
        default="CUDA",
        help="""Platform on which to simulation. Can be CUDA, CPU, or
        Reference."""
    )
    parser.add_argument(
        "-precision",
        nargs="?",
        const="mixed",
        default="mixed",
        help="""Floating point precision mode. Mixed assembles forces in
        single precision, but integrates in double precision."""
    )
    parser.add_argument(
        "-minimisation_tolerance",
        nargs="?",
        type=float,
        const=1.0,
        help="""Tolerance threshold for energy minimiser in kilojoule/mol.""",
        default=1.0
    )
    parser.add_argument(
        "-minimisation_steps",
        nargs="?",
        type=int,
        const=1000,  # required even for umbrella sampling
        help="""Maximum number of energy minimsation steps.""",
        default=1000
    )
    parser.add_argument(
        "-equilibration_steps",
        nargs=None,
        type=int,
        default=1000,
        help="""Number of equilibration steps with reduced time step prior
        to productive sampling."""
    )
    parser.add_argument(
        "-polarisation",
        nargs="?",
        const="extrapolated",
        default="extrapolated",
        help="""Method for calculating induced dipoles. Can be mutual,
        extrapolated, or direct."""
    )
    parser.add_argument(
        "-mutual_induced_target_epsilon",
        type=float,
        nargs="?",
        const=1e-5,
        default=1e-5,
        help="""Relative error tolerance for coverging induced dipoles in
        self-consistent field (mutual) approach."""
    )
    parser.add_argument(
        "-nonbonded_cutoff",
        type=float,
        nargs="?",
        const=0.8,        # value used in AMOEBA parameterisation
        default=0.8,
        help="""Cutoff for all non-bonded forces except van der Waals in
        nm."""
    )
    parser.add_argument(
        "-vdw_cutoff",
        type=float,
        nargs="?",
        const=1.2,        # value used in AMOEBA parameterisation
        default=1.2,
        help="""Cutoff for van der Waals forces in nm."""
    )
    parser.add_argument(
        "-ewald_error_tolerance",
        type=float,
        nargs="?",
        const=5e-4,        # the default in openMM
        default=5e-4,
        help="""Relative tolerance for PME calculations."""
    )
    parser.add_argument(
        "-integrator",
        nargs="?",
        const="mts",
        default="mts",
        help="""Time stepping algorithm. Can be verlet or mts (RESPA)."""
    )
    parser.add_argument(
        "-timestep",
        type=float,
        nargs=None,
        default=2.0,
        help="""Time step in femtoseconds. For MTS integrator this is the
        outer time step."""
    )
    parser.add_argument(
        "-inner_ts_frac",
        type=int,
        nargs=None,
        default=8,
        help="""Inner time step as fraction of out time step. Ignored for
        Verlet integrator."""
    )
    parser.add_argument(
        "-num_steps",
        type=int,
        nargs="?",
        const=100,
        default=100,
        help="""Number of time steps to compute."""
    )
    parser.add_argument(
        "-report_freq",
        type=int,
        nargs="?",
        const=25,
        default=25,
        help="""Write output every report_freq steps."""
    )
    parser.add_argument(
        "-temperature",
        type=float,
        nargs="?",
        const=310,
        default=310,
        help="""Temperature in Kelvin."""
    )
    parser.add_argument(
        "-pressure",
        type=float,
        nargs="?",
        const=1,
        default=1,
        help="""Pressure in bar."""
    )
    parser.add_argument(
        "-restr",
        type=str,
        nargs="?",
        const="capr",
        default="capr",
        help="""Group of atoms to apply harmonic positional restraints to.
        Can be capr, hapr, or none."""
    )
    parser.add_argument(
        "-restr_fc",
        type=float,
        nargs="?",
        const=1000,
        default=1000,
        help="""Force constant for harmonic positional restraints in
        kilojoule/mol/nm^2."""
    )
    args = parser.parse_args()
    argdict = vars(args)

    # pass arguments to main function:
    main(argdict)
