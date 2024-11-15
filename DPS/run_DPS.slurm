#!/bin/bash
#SBATCH --job-name=DPS_Parallel_Run
#SBATCH --output=data/slurm/slurm_output_%A_%a.out
#SBATCH --error=data/slurm/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1              # 1 task per array job
#SBATCH --array=1-200           # Default array range, adjustable via num_instances

# Source Conda's initialization script
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust this path if needed

# Activate the Conda environment
conda activate SLP

# Ensure the data and slurm directories exist
mkdir -p data/slurm

# Get the total number of instances from the command line
if [ -z "$1" ]; then
    echo "Error: Please provide the number of instances as an argument."
    exit 1
fi

num_instances=$1

# Adjust the array size dynamically based on num_instances
if [ "$SLURM_ARRAY_TASK_ID" -gt "$num_instances" ]; then
    echo "Task ID exceeds the specified number of instances. Skipping."
    exit 0
fi

# Each task uses its unique array index (SLURM_ARRAY_TASK_ID) as the seed
seed="$SLURM_ARRAY_TASK_ID"

# Call DPS.py with the unique seed and a default NFE of 10000
python DPS.py --seed "$seed" --NFE 10000
