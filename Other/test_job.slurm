#!/bin/bash

#SBATCH --job-name test_jesus
#SBATCH --output Other/output_%j.txt
#SBATCH --error Other/errors_%j.txt
#SBATCH --mail-user albi25.zanca@gmail.com
#SBATCH --mail-type END,FAIL

#SBATCH --ntasks 16
#SBATCH --partition allgroups
#SBATCH --time 5:00
#SBATCH --mem 2G

cd $SLURM_SUBMIT_DIR

ls -la

srun singularity exec library://jesus-333/collection/base_conda_numba python AoI_TDMA_Simulation.py
