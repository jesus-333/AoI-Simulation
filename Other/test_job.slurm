#!/bin/bash
#SBATCH --ntasks 3
#SBATCH --partition allgroups
#SBATCH --time 5:00
#SBATCH --mem 1G

cd $SLURM_SUBMIT_DIR

srun python AoI_TDMA_simulation.py
