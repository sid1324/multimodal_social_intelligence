#!/bin/bash
#SBATCH -A cis260338p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
# Untrained baselines: base model with the R0 (social) prompt and with the R1 (spatial) prompt.
set -euo pipefail
cd $PROJECT/rl_pilot
module load anaconda3/2024.10-1
source $PROJECT/idea5_pilot/venv/bin/activate
export HF_HOME=$PROJECT/idea5_pilot/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[ -f results/base_social.json ] || python eval_rl.py --format social --adapter base --out results/base_social.json
[ -f results/base_spatial.json ] || python eval_rl.py --format spatial --adapter base --out results/base_spatial.json
echo BASE_DONE
