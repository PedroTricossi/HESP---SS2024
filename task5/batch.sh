#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --job-name=particle_simulation
#SBATCH --export=NONE
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

unset SLURM_EXPORT_ENV
module load load nvhpc cuda
./run 100 0.1 32768 1 1 300 2.5 2 6.4 1 
