#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --job-name amoebajob
#SBATCH -p small

# load libraries and conda environment
module load cuda/8.0
module load python3/anaconda
source activate openmm-7-1-1

# bookkeeping
echo "CUDA visible devices: " $CUDA_VISIBLE_DEVICES
nvidia-smi > smi
echo "Starting MD run: $(date)"

# run simulation from restart file
python simulate_amoeba.py \
    -log production_parameters.log

# clean up workspace
source deactivate
module purge
