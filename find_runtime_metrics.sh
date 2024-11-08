#!/bin/bash

# Number of seeds
NSEEDS=1
SEEDS=$(seq 1 ${NSEEDS})

# Path to MOEAFramework JAR file
JAVA_ARGS="-cp MOEAFramework-4.4-Demo.jar"

# Ensure output directories exist
mkdir -p DPS/metrics
mkdir -p DPS/objs
mkdir -p output
mkdir -p error

# Loop through each seed and submit a Slurm job
for SEED in ${SEEDS}
do
	# Define Slurm job name and output paths
	NAME=Runtime_Metrics_S${SEED}
	OBJ_FILE="./DPS/objs/LakeDPS_S${SEED}.obj"      # Path to objectives file
	METRICS_FILE="./DPS/metrics/LakeDPS_S${SEED}.metrics" # Output metrics file
	REFERENCE_FILE="Overall.reference"               # Path to reference set file

	# Submit job to Slurm
	sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}                  # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --time=1:00:00                      # Walltime
#SBATCH --output=output/${NAME}.out         # Standard output
#SBATCH --error=error/${NAME}.err           # Error output

# Navigate to working directory
cd \$SLURM_SUBMIT_DIR

# Run MOEAFramework ResultFileEvaluator for the current seed
java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.ResultFileEvaluator \
	-d 4 -i ${OBJ_FILE} -r ${REFERENCE_FILE} -o ${METRICS_FILE}
EOF

done
