#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Concatenates output of OpenMM Amoeba simulations.

This script loads all CSV files whose name agrees with the pattern
[basename]_*.csv from the current directory and uses pandas to concatenate
them into one CSV file names [basename].csv, where [basename] is given by the
user. This does not check for duplicate timesteps or the like, so the user
must take care of this before calling the script. It does however sort the
entries in the CSV file by the value of the time step column.

The script also loads all DCD files whose name agrees with the pattern
[basename]_*.dcd from the current directory into an MDAnalysis universe. The
trajectories are then written to one single output XTC file with the name
[basename].xtc. Again, not much sanity checking is performed outside of what
MDAnalysis does internally.
"""

import argparse
import glob
import warnings

import pandas as pd

import MDAnalysis as mda


def main(argdict):
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # find all CSV files in with given basename in this directory:
    csv_files = glob.glob(args.basename + "_*.csv")

    # load all csv files:
    csv_data = [pd.read_csv(csv_file, delimiter="\t") for csv_file in csv_files]

    # concatenate into single data frame:
    csv_data = pd.concat(csv_data)

    # sort by second column (should contain time stamps):
    csv_data = csv_data.sort_values('Time (ps)')

    # write to overall CSV file:
    csv_data.to_csv(args.basename + ".csv", sep="\t", index=False)

    # find all DCD files in with given basename in this directory:
    dcd_files = glob.glob(args.basename + "_*.dcd")

    # sort DCD files by restart number:
    restart_number = [
        int(dcd_file.split("_")[-1].split(".dcd")[0]) for dcd_file in dcd_files
    ]
    sorted_dcd_files = [x for _, x in sorted(zip(restart_number, dcd_files))]

    # load trajectories into an MDAnalsysis universe:
    u = mda.Universe(args.basename + ".pdb", sorted_dcd_files)

    # select all particles:
    system = u.select_atoms("all")

    # write to XTC trajectory:
    prev_time = u.trajectory.time - u.trajectory.dt
    with mda.Writer(args.basename + ".xtc", system.n_atoms) as w:
        for ts in u.trajectory:

            print(ts)

            # advance time step:
            ts.time = prev_time + ts.dt

            # check for blowups:
            if (
                not min(ts.positions.flatten()) <= -1000
                and not max(ts.positions.flatten()) >= 9999.999
            ):
                w.write(system)
            else:
                warnings.warn("Possible blow up, skipping this frame.")

            # advance time step:
            prev_time = ts.time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-basename",
        type=str,
        nargs="?",
        const="amoeba",
        default="amoeba",
        help="Basename of output file without file extension or restart number."
    )
    args = parser.parse_args()
    argdict = vars(args)

    main(argdict)
