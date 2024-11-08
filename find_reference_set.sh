#!/bin/bash

# Paths
SET_FILES_PATH="./DPS/sets/*.set"        # Path to your .set files
PARETO_PY="./pareto.py"                  # Path to pareto.py
DPS_RESULT_FILE="DPS.resultfile"         # Intermediate result file for DPS
REFERENCE_FILE="DPS.reference"           # Final DPS reference file
OVERALL_REFERENCE="Overall.reference"    # Final combined reference file

# Step 1: Preprocess .set files to extract only the Objectives
# This will create temporary files with only the objectives values in a clean format
for set_file in ${SET_FILES_PATH}
do
    # Extract lines containing "Objectives:", remove text labels, and format only numeric data
    grep "Objectives" "${set_file}" \
    | sed 's/.*Objectives: \[//' \
    | sed 's/\].*//' \
    | tr -d ',' \
    | tr -s ' ' > "${set_file}.obj"
done

# Step 2: Run Pareto analysis on the preprocessed .obj files
python ${PARETO_PY} ./DPS/sets/*.obj -o 0-3 -e 0.01 0.01 0.001 0.001 --output ${DPS_RESULT_FILE} --delimiter=" " --comment="#"

# Step 3: Process DPS.resultfile to create DPS.reference
cut -d ' ' -f 1-4 ${DPS_RESULT_FILE} > ${REFERENCE_FILE}

# Step 4: Create the final overall reference set
python ${PARETO_PY} ./${REFERENCE_FILE} -o 0-3 -e 0.01 0.01 0.001 0.001 --output ${OVERALL_REFERENCE} --delimiter=" " --comment="#"

# Optional: Clean up temporary .obj files if needed
rm ./DPS/sets/*.obj
