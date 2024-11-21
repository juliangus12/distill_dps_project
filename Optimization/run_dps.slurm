#!/bin/bash
#SBATCH --job-name=DPS_Parallel_Run
#SBATCH --output=slurm_output/slurm_output_%A_%a.out
#SBATCH --error=slurm_output/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-50

mkdir -p slurm_output # Directory for slurm output and error files 
mkdir -p solutions # Directory for sotring cache files containing RBF parameters and objective values 
mkdir -p pareto_frontiers
# Run the DPS optimization script
python dps.py
