#!/bin/sh
#
#SBATCH --account=dsi        # Replace ACCOUNT with your group account name
#SBATCH --job-name=E5Server    # The job name
#SBATCH -c 32                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-6:00            # The time the job will take to run in D-HH:MM
#SBATCH --gres=gpu:2
#SBATCH --output=e5_output.log
#SBATCH --mem=50GB

hostname -i | awk '{print $1}' > ip_e5.txt # print out the hostname
cd /burg/home/jef2182/LMOps/corag/
source .venv/bin/activate
cd /burg/home/jef2182/smart-search/corag/
PYTHONPATH=src/ uvicorn src.search.start_e5_server_main:app --host 0.0.0.0 --port 8090 --forwarded-allow-ips="*"

# End of script
