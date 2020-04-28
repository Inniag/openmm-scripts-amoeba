*Please cite **G. Klesse, S. Rao, S. J. Tucker, and M. S. P. Sansom. "Induced polarization in MD simulations of the 5HT3 receptor channel." Journal of the American Chemical Society, 2020.** if you make use of these scripts in your academic work.*

# OpenMM Scripts for AMOEBA Force Field MD Simulations

This repository provides scripts for running GPU-accelerated molecular dynamics simulations employing the polarisable [AMOEBA](https://doi.org/10.1021/jp910674d) force field. The code is geared towards the specific use case of simulating ion channels, but should also be useful for other membrane proteins as well as soluble biomolecules. It makes use of the powerful [OpenMM](https://github.com/openmm/openmm) and [MDAnalysis](https://github.com/MDAnalysis/mdanalysis) libraries.

Some interesting features of this code that extend the basic OpenMM functionality include:

* guessing bonds in nonstandard residues which would overflow the CONECT record index in PDB files, a prerequisite for lipid bilayer simulation
* converting systems equilibrated in Gromacs to valid OpenMM input and converting OpenMM DCD trajectories to XTC format enabling further analysis with [Gromacs](https://gitlab.com/gromacs/gromacs)' tools
* lipid parameters for the AMOEBA force field and a script for converting force field parameter files in [Tinker](https://dasher.wustl.edu/tinker/)'s PRM format to the XML format used by OpenMM
* performing umbrella sampling simulations of ions moving along the permeation pathway of an ion channel with output compatible with the Grossfield lab's [WHAM](http://membrane.urmc.rochester.edu/?page_id=126) tool
* job submission scripts for running simulations on [SLURM](https://slurm.schedmd.com/documentation.html) clusters and support for restarting and extending simulations that exceed queuing system time limits

The code is structured into self-contained Python scripts. This is intended to enable easy portability to support the common scientific workflow of preparing simulations on a local workstation, executing them on a remote HPC cluster, and analysing data back on the workstation.


## Dependencies

The scripts in this repository depend on both OpenMM and MDAnalysis as well as on a number of underlying packages from the Python scientific stack. These dependencies are most easily installed using conda:

```
conda env create -f requirements.yml
conda activate openmm-scripts-amoeba
```

Note that this will install OpenMM version 7.1.1 as used in the JACS paper cited above, which only supports CUDA version 8.0. Newer versions of OpenMM support different CUDA versions and you might want to adapt the `requirements.yml` file accordingly if you want to use a more recent CUDA version.
