#!/bin/sh
#
#SBATCH --account=dsi        # Replace ACCOUNT with your group account name
#SBATCH --job-name=CoRAG    # The job name
#SBATCH -c 32                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-4:00            # The time the job will take to run in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --output=corag.log
#SBATCH --mem=100GB

hostname -i # print out the hostname
cd /burg/home/jef2182/LMOps/corag/
source .venv/bin/activate
cd /burg/home/jef2182/smart-search/corag/
# bash scripts/eval_multihopqa.sh
PYTHONPATH=src/ python src/inference/run_inference.py \
        --eval_task "hotpotqa" \
        --eval_split "validation" \
        --max_path_length "6" \
        --decode_strategy "tree_search" \
        --output_dir "tmp/6" \
        --do_eval \
        --num_threads 32 \
        --overwrite_output_dir \
        --disable_tqdm True \
        --report_to none "$@" \
        --vllm_ip "$VLLM_IP" \
        --e5_ip "$E5_IP"
#        --dry_run True
# End of script
