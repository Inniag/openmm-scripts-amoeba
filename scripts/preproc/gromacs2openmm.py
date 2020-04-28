#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepares PDB file for OpenMM AMOEBA simulation and deletes selected waters.

This script converts a GRO format structure file (e.g. resulting from
preequilibrating a system via atomistic simulation in Gromacs) to PDB format
using the MDAnalysis library. Note that PDB files written by many other common
tools (e.g. VMD) do not comply with the official PDB standard and are thus not
accepted by OpenMM.

As part of the conversion process, the script also deletes a set of molecules by
residue ID. This is useful for removing water molecules that have been trapped
somewhere inside the protein or at the protein-lipid interface during the setup
process and that have been unable to move during the equilibration run.
"""

import argparse

import MDAnalysis as mda


def main(argict):
    """Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # load GRO file:
    u = mda.Universe(str(args.filename) + ".gro")

    # select atoms and save to PDB file:
    if len(args.exclid) > 0:
        sys = u.select_atoms("not resid " + ' '.join(map(str, args.exclid)))
    else:
        sys = u.select_atoms("all")

    sys.write(str(args.filename) + ".pdb")


if __name__ == "__main__":

    # parse command line arguments:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-filename",
        nargs="?",
        const="equilibration",
        default="equilibration"
    )
    parser.add_argument(
        "-exclid",
        nargs="+",
        default=""
    )

    # parse arguments and convert namespace to dictionary:
    args = parser.parse_args()
    argdict = vars(args)

    # pass arguments to main function:
    main(argdict)
