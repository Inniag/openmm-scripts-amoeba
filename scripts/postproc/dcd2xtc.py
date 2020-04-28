#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convertes a DCD format trajectory into an XTC format trajectory.

Takes as its input a DCD format trajectory and a corresponding structure in PDB
format and returns the same trajectory in XTC format, making it accessible for
further post-processing using Gromacs tools.
"""

import argparse
import os

import MDAnalysis as mda


def main(argdict):
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # get basename of input trajectory:
    (basename, extension) = os.path.splitext(args.trajectory)

    # load trajectories into an MDAnalsysis universe:
    u = mda.Universe(args.structure, args.trajectory)

    # select all particles:
    system = u.select_atoms("all")

    # write to XTC trajectory:
    with mda.Writer(basename + ".xtc", system.n_atoms) as w:
        for ts in u.trajectory:
            w.write(system)


if __name__ == "__main__":

    # parse command line arguments:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-structure",
        type=str,
        nargs="?",
        const="atomistic_system_openmm.pdb",
        default="amoeba",
        help="Full name of input structure file."
    )
    parser.add_argument(
        "-trajectory",
        type=str,
        nargs="?",
        const="production.dcd",
        default="production.dcd",
        help="Full name of input trajectory."
    )
    args = parser.parse_args()
    argdict = vars(args)

    main(argdict)
