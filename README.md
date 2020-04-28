Please cite *G. Klesse, S. Rao, S. J. Tucker, and M. S. P. Sansom. "Induced polarization in MD simulations of the 5HT3 receptor channel." bioRxiv, 2020. Accepted for publication in the Journal of the American Chemical Society.* if you make use of these scripts in your academic work.

# OpenMM Scripts for AMOEBA Force Field MD Simulations

This repository provides scripts for running GPU-accelerated molecular dynamics simulations employing the polarisable AMOEBA force field. It makes use of the powerful OpenMM and MDAnalysis libraries. The code is geared towards the specific use case of simulating ion channels, but should also be useful for other membrane proteins as well as soluble biomolecules.

Some interesting features of this code that extend the basic OpenMM functionality include:

* guessing bonds in nonstandard residues which would overflow the CONECT record index in PDB files, a prerequisite for lipid bilayer simulation
* converting systems equilibrated in Gromacs to valid OpenMM input and converting OpenMM DCD trajectories to XTC format for analysis with Gromacs tools
* lipid parameters for the AMOEBA force field and a script for converting force field parameter files in Tinker's PRM format to the XML format used by OpenMM
* performing umbrella sampling simulations of ions moving along the permeation pathway of an ion channel with output compatible with the Grossfield lab's WHAM tool
* job submission scripts for running simulations on SLURM clusters and support for restarting and extending simulations that exceed queuing system time limits

The code is structured into self-contained Python scripts. This is intended to enable easy portability to support the common scientific workflow of preparing simulations on a local workstation, executing them on a remote HPC cluster, and analysing data back on the workstation.
