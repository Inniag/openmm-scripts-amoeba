# Pre-processing Scripts for OpenMM AMOEBA Simulations

This directory provides a number of scripts helpful in setting up AMOEBA force field simulations with OpenMM.

* `gromacs2openmm.py` converts coordinate files from Gromacs' GRO format to the PDB format used by OpenMM
* `prepare_umbrella_windows.py` sets up a set of simulation systems for umbrella sampling along an ion channel permeation pathway
* `tinker2openmm.py` converts force field parameters from Tinker's PRM format to the XML format used by OpenMM

Type `./scriptname -h` to see a help on which input parameters need to be provided.
