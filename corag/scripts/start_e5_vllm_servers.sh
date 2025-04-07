#!/bin/bash

# E5 Job
# Submit the batch job and capture the Job ID
JOB_ID=$(sbatch --parsable /burg/home/jef2182/smart-search/corag/scripts/e5_job.sh)
echo "Submitted job with ID: $JOB_ID"

# Wait for the job to start and get the allocated node
echo "Waiting for job to start..."
while :; do
    COMPUTE_NODE=$(squeue --job $JOB_ID --noheader --format=%N)
    if [[ -n "$COMPUTE_NODE" && "$COMPUTE_NODE" != "(null)" ]]; then
        break
    fi
    sleep 2  # Check every 2 seconds
done

# Wait for the file to be created
while [ ! -f ip_e5.txt ]; do
    sleep 1
done

# Get the IP of the compute node
export E5_IP=$(cat ip_e5.txt)
echo "E5 Server Compute node allocated: $COMPUTE_NODE ($E5_IP)"
rm -f ip_e5.txt # cleanup

# VLLM Job
# Submit the batch job and capture the Job ID
JOB_ID=$(sbatch --parsable /burg/home/jef2182/smart-search/corag/scripts/vllm_job.sh)
echo "Submitted job with ID: $JOB_ID"

# Wait for the job to start and get the allocated node
echo "Waiting for job to start..."
while :; do
    COMPUTE_NODE=$(squeue --job $JOB_ID --noheader --format=%N)
    if [[ -n "$COMPUTE_NODE" && "$COMPUTE_NODE" != "(null)" ]]; then
        break
    fi
    sleep 2  # Check every 2 seconds
done

# Wait for the file to be created
while [ ! -f ip_vllm.txt ]; do
    sleep 1
done

# Get the IP of the compute node
export VLLM_IP=$(cat ip_vllm.txt)
echo "VLLM Server Compute node allocated: $COMPUTE_NODE ($VLLM_IP)"
rm -f ip_vllm.txt # cleanup
