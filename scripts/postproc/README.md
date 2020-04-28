# Post-processing Scripts for OpenMM AMOEBA Simulations

This directory provides a number of scripts useful for analysing data resulting from AMOEBA force field simulations in OpenMM.

* `concat_amoeba.py` combines the output of several successive OpenMM simulations into one XTC trajectory file and one state data CSV file
* `dcd2xtc` converts an individual DCD format trajectory into an XTC format trajectory facilitating further processing with Gromacs tools
* `dipole_profile_amoeba.py` calculates the molecular dipole moment of water along the permeation pathway of an ion channel as determined by [CHAP](www.channotation.org)
* `plot_thermodynamics_properties.R` plots thermodynamic state variables over time; useful for checking whether a simulation is equilibrated and stable

Type `./scriptname -h` to see a help on which input parameters need to be provided.
