#!/bin/bash

#SBATCH --job-name test_jesus
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user albi25.zanca@gmail.com
#SBATCH --mail-type END,FAIL

#SBATCH --ntasks 3
#SBATCH --partition allgroups
#SBATCH --time 5:00
#SBATCH --mem 1G

cd $SLURM_SUBMIT_DIR

ls -la

srun python AoI_TDMA_simulation.py
