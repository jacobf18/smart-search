#!/bin/sh
#
#SBATCH --account=dsi        # Replace ACCOUNT with your group account name
#SBATCH --job-name=CoRAG    # The job name
#SBATCH -c 8                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-12:00            # The time the job will take to run in D-HH:MM
#SBATCH --output=corag.log
#SBATCH --mem=20GB

hostname -i # print out the hostname
source /burg/dsi/users/ds4386/envs/smart_search/bin/activate
cd /burg/home/jef2182/smart-search/corag/
# bash scripts/eval_multihopqa.sh
PYTHONPATH=src/ python src/inference/run_inference.py \
        --eval_task "hotpotqa" \
        --eval_split "validation" \
        --max_path_length "10" \
        --decode_strategy "phs" \
        --output_dir "tmp/10/pass_5" \
        --do_eval \
        --num_threads 1 \
        --overwrite_output_dir \
        --disable_tqdm True \
        --report_to none "$@" \
        --vllm_ip "$VLLM_IP" \
        --e5_ip "$E5_IP"
        # --dry_run True
        # --best_n 8 
# End of script
