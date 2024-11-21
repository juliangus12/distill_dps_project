#!/bin/bash
#SBATCH --job-name=Distill_Parallel_Run
#SBATCH --output=slurm_output/slurm_output_%A_%a.out
#SBATCH --error=slurm_output/slurm_error_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-50

# Create directories for output
mkdir -p slurm_output
mkdir -p distill_output

# Define the directory containing Pareto frontier files
pareto_dir="../Optimization/pareto_frontiers"

# Convert the list of files into an array
pareto_files=("$pareto_dir"/*)

# Get the index of the current task
index=$((SLURM_ARRAY_TASK_ID - 1))

# Check if the index is within bounds
if [ $index -ge 0 ] && [ $index -lt ${#pareto_files[@]} ]; then
    # Get the Pareto file corresponding to this task
    pareto_file="${pareto_files[$index]}"
    
    # Extract the base filename (without extension) for unique output naming
    base_filename=$(basename "$pareto_file")
    unique_id="${base_filename%.*}"

    # Run distill.py with the current Pareto file and specify output directory
    python distill.py --pareto-file "$pareto_file" --scratch-dir "distill_output/dagger_scratch_${unique_id}_${SLURM_ARRAY_TASK_ID}"
else
    echo "Index $index out of bounds for SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"
fi
