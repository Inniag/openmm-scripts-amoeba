# Scripts for Running AMOEBA Force Field Simulations in OpenMM

This directory provides scripts for carrying out GPU-accelerated MD simulations using the AMOEBA force field in OpenMM.

* `simulate_amoeba.py` performs an individual equilibrium molecular dynamics simulation
* `simulate_amoeba_umbrella` performs an MD simulation where the relative position of a target ion is harmonically restrained, facilitating umbrella sampling

Type `./scriptname -h` to see a help on which input parameters need to be provided.

For examples of how to run simulations with these scripts on a remote cluster see my [SLURM submit scripts](../../scripts/cluster-submission/README.md).

Both scripts write the simulation parameters to a log file and provide a mechanism for extending simulations from checkpoint files, see the `-log` flag. This is especially helpful when running long simulations that can not be completed within the time limit set by a cluster queuing system. A tool for combining the output of several such extended simulations into a single trajectory can be found amongst the [post-processing scripts](../postproc/README.md).

A tool for setting up multiple umbrella windows is also available as a [pre-processing scripts](../postproc/README.md)
