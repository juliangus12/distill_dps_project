#!/bin/bash
#SBATCH --job-name=DPS_Parallel_Run
#SBATCH --output=slurm_output/slurm_output_%A_%a.out
#SBATCH --error=slurm_output/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-200

# Ensure necessary directories exist
mkdir -p slurm_output
mkdir -p caches

# Run the DPS optimization script
python dps.py
