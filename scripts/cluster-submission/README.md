# Example Scripts for Cluster Submission

These scrips illustrate how to submit polarisable OpenMM jobs to an HPC cluster's queueing system. They were tried and tested on the UK's [JADE](https://www.jade.ac.uk/) cluster, but should be straightforward to adapt to any cluster using the Slurm workload manager.

* `submit_openmm_jade_1GPU.sh` submits a new job starting from fresh input files
* `extend_openmm_jade_1GPU.sh` submits a job that extends a previous simulation

When submitting jobs, make sure that all required input files (notably the input geometry file `system.pdb` and all necessary force field files, e.g. `amoeba2013.xml`, `dopc.xml`, etc.) are present in the working directory on the cluster.
