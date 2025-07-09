#!/bin/bash

# Your sbatch script filename
SBATCH_SCRIPT="train_network.sh"
CONFIG_FILE=$1  # grab first argument to this script

# Submit the first job and capture its job ID
job_id=$(sbatch $SBATCH_SCRIPT "$CONFIG_FILE" | awk '{print $4}')
echo "Submitted job with ID: $job_id using config $CONFIG_FILE"

for i in {2..3}
do
    job_id=$(sbatch --dependency=afterany:$job_id $SBATCH_SCRIPT "$CONFIG_FILE" | awk '{print $4}')
    echo "Submitted job $i with ID: $job_id (afterany dependency)"
done