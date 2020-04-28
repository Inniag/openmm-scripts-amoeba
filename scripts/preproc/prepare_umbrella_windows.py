#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sets up umbrella windows for OpenMM AMOEBA FF simulation.

This script prepares windows for umbrella sampling simulations of an ion passing
through an ion channel protein. The user needs to provide a PDB and OpenMM XML
state file file containing a pre-equilibrated simulation system as well as a
resource directory containing all files (e.g. topologies) required for
simulation. This script will then create a directory of sequentially numbered
window directories, each of which will contain a modified PDB file of the
simulation system, where the target particle has been placed at a z-position
corresponding to the value of the collective variable in that window.

The script assumes that the input PDB contains a single ion channel protein
aligned with the Cartesian z-coordinate as well as an ionic solution and
(possibly) a lipid bilayer. The reaction coordinate / collective variable is
taken to be the z-coordinate and it is assumed that the ion conduction pathway
passes through the center of the protein. A user-specified target particle is
then placed at successive positions along the reaction coordinate and its x-
and y-coordinates are set to the center of geometry of the protein. The
z-coordinate covers the bounding box of the protein plus some user-specified
margin, but the script will never place the particle beyond the periodic box
boundary. The length of the z-interval is also user-specified.

The script will look for small molecules (e.g. water, ions) near the target
location and will attempt to swap these particles with target particle. This
helps to avoid clashes, but a few steps of energy minimisation prior to MD
simulation may still be necessary. If no swapping candidate can be found near
the target location, the target particle is simply placed exactly at the target
location.
"""


import argparse
import json
import os
import shutil
import xml.etree.ElementTree as et

import numpy as np

import MDAnalysis as mda


def box_dimensions_from_xml(xmlfile):
    """Reads periodic box dimensions from an OpenMM XML state file.

    Box dimensions are returned as a dimensions vector as understood by
    MDAnalysis.

    This function is necessary as the PDB files written by OpenMM contain the
    box vectors of the initial configuration of a simulation rather than the
    appropriate box vectors for the coordinates of that frame. Using the wrong
    box vectors will cause simulation crashes, so patching the PDB box vectors
    with the (accurate) box vectors from the XML file also written by OpenMM
    is necessary.
    """

    # parse XML file:
    tree = et.parse(xmlfile)
    root = tree.getroot()

    # create dimension array from XML data (assume rectangular box):
    nm2ang = 10.0   # unit conversion factor
    dimensions = np.array([
        float(root.find("PeriodicBoxVectors").find("A").attrib["x"]) * nm2ang,
        float(root.find("PeriodicBoxVectors").find("B").attrib["y"]) * nm2ang,
        float(root.find("PeriodicBoxVectors").find("C").attrib["z"]) * nm2ang,
        90.0,
        90.0,
        90.0
    ])

    # return dimensions to caller:
    return(dimensions)


def main(argict):
    """Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # load pre-equilibrated template structure:
    template_structure = os.path.abspath(argdict["f"])
    u = mda.Universe(
        template_structure,
        in_memory=True
    )

    # correct periodic box size:
    u.coord.dimensions = box_dimensions_from_xml(argdict["xml"])

    # create selection for overall system and for ion channel:
    syst = u.select_atoms("all")
    prot = u.select_atoms("protein")

    # find first atom that matches user-defined name, then identify bonded atoms
    umbrella_target = syst.select_atoms("name " + argdict["target_name"])
    target_segid = str(umbrella_target[0].segid)
    target_resid = str(umbrella_target[0].resid)
    target_name = str(umbrella_target[0].name)
    target_sel_string = ("atom "
                         + target_segid + " "
                         + target_resid + " "
                         + target_name
                         + " or (bonded atom "
                         + target_segid + " "
                         + target_resid + " "
                         + target_name + ")"
                         )
    umbrella_target = syst.select_atoms(target_sel_string)
    print("umbrella target: " + str(umbrella_target))

    # get original position of target particle:
    target_orig_pos = umbrella_target.center_of_geometry()
    print("original position: " + str(target_orig_pos))

    # create dictionary identifying the umbrella target:
    umbrella_target_dict = {
        "target_particle": {
            "altLoc": str(umbrella_target[0].altLoc),
            "id": int(umbrella_target[0].id),
            "index": int(umbrella_target[0].index),
            "name": str(umbrella_target[0].name),
            "occupancy": float(umbrella_target[0].occupancy),
            "resid": int(umbrella_target[0].resid),
            "resindex": int(umbrella_target[0].resindex),
            "resname": str(umbrella_target[0].resname),
            "resnum": int(umbrella_target[0].resnum),
            "segid": str(umbrella_target[0].segid),
            "segindex": int(umbrella_target[0].segindex),
            "tempfactor": float(umbrella_target[0].tempfactor),
            "type": str(umbrella_target[0].type)
        },
        "umbrella_params": {
            "cv": float(0.0),    # will be changed in loop below
            "fc": argdict["fc"],  # must be in kJ/mol/nm^2 for OpenMM
        },
        "setup_argdict": argdict
    }

    # extract command line parameters and convert units:
    nm2ang = 10.0   # unit conversion factor
    ang2nm = 0.1
    delta_z = argdict["delta_z"] * nm2ang
    z_mar_prot = argdict["z_mar_prot"] * nm2ang
    z_mar_box = argdict["z_mar_box"] * nm2ang
    window_basepath = argdict["window_dir"]

    # select limits of CV sufficiently far from protein, but make sure they are
    # inside the periodic box boundaries:
    z_min_prot = prot.bbox()[0, 2]
    z_max_prot = prot.bbox()[1, 2]
    z_min_syst = syst.bbox()[0, 2]
    z_max_syst = syst.bbox()[1, 2]
    z_min = max(z_min_prot - z_mar_prot, z_min_syst + z_mar_box)
    z_max = min(z_max_prot + z_mar_prot, z_max_syst - z_mar_box)

    # calculate values of collective variable (z-coordinate):
    z_tar = np.linspace(
        np.ceil(z_min / delta_z) * delta_z,
        np.floor(z_max / delta_z) * delta_z,
        np.abs(z_max - z_min) / delta_z
    )

    # define x/y coordinates of target particle initial placement:
    x_tar = prot.center_of_geometry()[0]
    y_tar = prot.center_of_geometry()[1]

    # loop over z-value:
    for i in range(0, len(z_tar)):

        # inform user:
        print("\nsetting up window " + str(i) + " at z = " + str(z_tar[i]))

        # set collective variable:
        # (note conversion from Ang to nm)
        umbrella_target_dict["umbrella_params"]["cv"] = float(z_tar[i] * ang2nm)

        # current and target position of umbrella target particle:
        pos_target = np.array([x_tar, y_tar, z_tar[i]])

        # find atoms in neighbourhood of target positions:
        clash_margin = argdict["clash_margin"] * nm2ang
        target_neighbours = syst.select_atoms(
            "point "
            + str(x_tar) + " "
            + str(y_tar) + " "
            + str(z_tar[i]) + " "
            + str(clash_margin)
            + " and name " + argdict["clash_species"]
            + " and not (" + target_sel_string + ")"  # avoid self-replacement
        )

        # did we find neighbouring atoms?
        if target_neighbours.n_atoms > 0:

            # find nearest neighbour:
            idx_nearest = np.argmin(
                np.linalg.norm(
                    target_neighbours.positions - pos_target,
                    axis=1
                )
            )
            nearest_atm = target_neighbours[idx_nearest]

            # guess bonds in case connectivity was not specified in PDB:
            res = syst.select_atoms("byres bynum " + str(nearest_atm.index + 1))
            res.guess_bonds(vdwradii={"D": 0.1})  # does not find Drude atoms

            # find all atoms bonded to nearest neighbour atom:
            # (simply byres statment does not work if PDB has crooked resindex)
            nearest_mol = syst.select_atoms(
                "not protein and ( "
                + "bynum " + str(nearest_atm.index + 1)
                + " or bonded bynum " + str(nearest_atm.index + 1)
                + " or bonded bonded bynum " + str(nearest_atm.index + 1)
                + " or (around 0.25 bynum " + str(nearest_atm.index + 1) + ")"
                + ")"
            )

            # inform user:
            print(
                "--> nearest atom: " + str(nearest_atm)
                + " IX: " + str(nearest_atm.ix)
                + " ID: " + str(nearest_atm.id)
                + " INDEX: " + str(nearest_atm.index)
            )
            print(
                "--> nearest mol: " + str(nearest_mol)
                + " number: " + str(nearest_mol.n_atoms)
            )

            # move umbrella target to position of nearest neighbour:
            umbrella_target.positions = (
                umbrella_target.positions
                - umbrella_target.center_of_geometry()
                + nearest_mol.center_of_geometry()
            )

            # move nearest neighbour to original position of target:
            nearest_mol.positions = (
                nearest_mol.positions
                - nearest_mol.center_of_geometry()
                + target_orig_pos
            )

        else:

            # simply move umbrella target to target position:
            umbrella_target.positions = (
                umbrella_target.positions
                - umbrella_target.center_of_geometry()
                + pos_target
            )

        # inform user:
        print(
            "--> new target position: " + umbrella_target.positions
        )

        # create new directory for this window:
        window_dir = window_basepath + "/window-" + format(i, "d")
        os.makedirs(window_dir)

        # write confirmation to PDB file:
        syst.write(
            os.path.join(
                window_dir,
                os.path.basename(argdict["umbrella_input_pdb"])
            )  # FIXME make file name command line parameter
        )

        # write the umbrella target dictionary to a JSON file:
        with open(str(window_dir) + "/umbrella_target.json", "w") as f:
            json.dump(umbrella_target_dict, f, sort_keys=True)

        # copy resources to this window:
        for src in os.listdir(os.path.abspath(argdict["resource_dir"])):
            if (
                os.path.isdir(
                    os.path.join(os.path.abspath(argdict["resource_dir"]), src)
                )
            ):
                shutil.copytree(
                    os.path.join(argdict["resource_dir"], src),
                    os.path.join(window_dir, src)
                )
            else:
                shutil.copy(
                    os.path.join(argdict["resource_dir"], src),
                    os.path.join(window_dir, src)
                )


# entry point check:
if __name__ == "__main__":

    # parse command line arguments:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        type=str,
        nargs=None,
        default="production.pdb",
        help="""Name of template structure. Should be equilibrated."""
    )
    parser.add_argument(
        "-xml",
        type=str,
        nargs=None,
        default="production.xml",
        help="""Name of OpenMM state file containing accurate box
        dimensions."""
    )
    parser.add_argument(
        "-target_name",
        type=str,
        nargs=None,
        default="SOD",
        help="""Name of atom to place under umbrella restraints. Will pick the
        first atom with this name in PDB file."""
    )
    parser.add_argument(
        "-resource_dir",
        type=str,
        nargs=None,
        default="resources",
        help="""Directory containing any additional files required for
        simulation, such as PSF topology or CHARMM stream files. The contents
        will be copied to each window directory."""
    )
    parser.add_argument(
        "-window_dir",
        type=str,
        nargs=None,
        default="windows",
        help="""Name of directory in which to place umbrella window starting
        configurtations."""
    )
    parser.add_argument(
        "-umbrella_input_pdb",
        type=str,
        nargs=None,
        default="umbrella_input.pdb",
        help="""Name of PDB file written as input for umbrella samling MD."""
    )
    parser.add_argument(
        "-fc",
        type=float,
        nargs=None,
        default=1000.0,
        help="""Force constant for umbrella potential in
        kilojoule/mol/nm^2."""
    )
    parser.add_argument(
        "-z_mar_prot",
        type=float,
        nargs=None,
        default=1.0,
        help="""Length in nm beyond protein bounding box over which to extend
        collective variable."""
    )
    parser.add_argument(
        "-z_mar_box",
        type=float,
        nargs=None,
        default=0.5,
        help="""Minimum distance between target particle and box boundary in
        nm. This is a heuristic to take into account box size fluctuations in
        the NPT ensemble."""
    )
    parser.add_argument(
        "-delta_z",
        type=float,
        nargs=None,
        default=0.1,
        help="""Distance in nm between two subsequent umbrella windows."""
    )
    parser.add_argument(
        "-clash_margin",
        type=float,
        nargs=None,
        default=0.3,
        help="""Distance margin in nm for detecting clashes."""
    )
    parser.add_argument(
        "-clash_species",
        type=str,
        nargs=None,
        default="OH2 SOD CLA",
        help="""Single string listing the atoms whith which the target
        particle can be swapped. Should be small species like ions or water."""
    )

    # parse arguments and convert namespace to dictionary:
    args = parser.parse_args()
    argdict = vars(args)

    # pass arguments to main function:
    main(argdict)
