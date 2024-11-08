#!/bin/bash

# Check for a single argument specifying the number of instances
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <number_of_instances>"
  exit 1
fi

# Number of instances to run
n=$1

# Directory to store output .set files
output_dir="./DPS/sets"
mkdir -p $output_dir

# Number of function evaluations for each instance
nfe=10000  # Adjust this value if needed

# Loop to submit each instance as a separate Slurm job
for ((seed=1; seed<=n; seed++))
do
  # Submit each job to Slurm
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=LakeProblem_${seed}
#SBATCH --output=${output_dir}/LakeProblem_${seed}.out
#SBATCH --error=${output_dir}/LakeProblem_${seed}.err
#SBATCH --time=01:00:00  # Adjust the time limit as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  # Adjust CPUs per task as needed
#SBATCH --mem=1G  # Adjust memory as needed

# Load any necessary modules or environment setup here
# module load python/3.x  # Uncomment and modify if needed

# Run the Python script with the current seed
python LakeProblem_borg.py --seed ${seed} --output-dir ${output_dir} --nfe ${nfe}
EOT

done

echo "Submitted ${n} jobs to Slurm, each with a unique seed."
