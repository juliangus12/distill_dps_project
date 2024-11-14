#!/bin/bash
#SBATCH --job-name=DPS_Run
#SBATCH --output=data/slurm_output_%A_%a.out  # %A is the job ID, %a is the array index
#SBATCH --error=data/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-200  # Define the job array size here

# Load necessary modules or activate the environment
module load python
source activate distill

# Make sure the data directory exists
mkdir -p data

# Use the Slurm task ID as the seed and pass it to DPS.py
python DPS.py "$SLURM_ARRAY_TASK_ID"
