#!/bin/bash
#SBATCH --job-name=DPS_Parallel_Run
#SBATCH --output=Optimization/slurm_output/slurm_output_%A_%a.out
#SBATCH --error=Optimization/slurm_output/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-200  # Adjust this range for the number of seeds

# Ensure the caches and slurm output directories exist
mkdir -p caches
mkdir -p slurm_output

# Execute DPS.py with the current SLURM_ARRAY_TASK_ID as the seed
python dps.py
