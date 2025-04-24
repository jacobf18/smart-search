#!/bin/sh
#
#SBATCH --account=dsi        # Replace ACCOUNT with your group account name
#SBATCH --job-name=VLLMServer    # The job name
#SBATCH -c 8                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-12:00            # The time the job will take to run in D-HH:MM
#SBATCH --gres=gpu:2
#SBATCH --output=vllm_client.log
#SBATCH --mem=50GB

hostname -i | awk '{print $1}' > ip_vllm.txt # print out the hostname
source /burg/dsi/users/ds4386/envs/smart_search/bin/activate
cd /burg/home/jef2182/smart-search/corag/
vllm serve corag/CoRAG-Llama3.1-8B-MultihopQA \
    --dtype auto \
    --disable-log-requests --disable-custom-all-reduce \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --enforce-eager
    #--dtype=half \
    # --api-key token-123

# End of script
