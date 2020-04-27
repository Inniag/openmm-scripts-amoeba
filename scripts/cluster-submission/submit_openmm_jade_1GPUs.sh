#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --job-name amoebajob
#SBATCH -p small
#SBATCH --mail-user=gianni.klesse@physics.ox.ac.uk

# load libraries and conda environment
module load cuda/8.0
module load python3/anaconda
source activate openmm-7-1-1

echo "CUDA visible devices: " $CUDA_VISIBLE_DEVICES
nvidia-smi > smi
echo "Starting MD run: $(date)"

# run simulation
python simulate_amoeba.py \
    -pdb system.pdb \
    -ff amoeba2013.xml dopc.xml \
    -outname production \
    -polarisation extrapolated \
    -num_steps 12500000 \
    -report_freq 5000 \
    -integrator mts \
    -timestep 2.0 \
    -inner_ts_frac 8

# clean up workspace
source deactivate
module purge
