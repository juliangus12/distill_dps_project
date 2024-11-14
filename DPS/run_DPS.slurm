#!/bin/bash
#SBATCH --job-name=DPS_Run
#SBATCH --output=data/slurm/slurm_output_%A_%a.out
#SBATCH --error=data/slurm/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-200

# Attempt direct activation
conda activate SLPv1

# Make sure the data directory exists
mkdir -p data

# Run your Python script
python DPS.py "$SLURM_ARRAY_TASK_ID"
