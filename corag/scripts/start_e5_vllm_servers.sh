#!/bin/bash

# E5 Job
# Submit the batch job and capture the Job ID
JOB_ID=$(sbatch --parsable e5_job.sh)
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
JOB_ID=$(sbatch --parsable vllm_job.sh)
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



# # Set up SSH tunnel to forward localhost:8090 to the compute node
# echo "Setting up SSH tunnel to $COMPUTE_NODE ($COMPUTE_IP:8090)"
# ssh -fN -L 8090:"$COMPUTE_IP":8090 $USER@$COMPUTE_NODE

# echo "Port forwarding is active. You can now access localhost:8090."
