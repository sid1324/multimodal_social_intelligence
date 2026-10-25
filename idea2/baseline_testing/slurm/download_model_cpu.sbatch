#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=02:00:00
#SBATCH --job-name=social-model-download
#SBATCH --output=logs/model-download-cpu-%j.log
#SBATCH --error=logs/model-download-cpu-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export SOCIAL_RL_STORE=${SOCIAL_RL_STORE:-/ocean/projects/cis260338p/apatawar/social_norm_rl}
export PYTHONPATH="$PWD/src"
export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$SOCIAL_RL_STORE/envs/data/bin/python" scripts/download_model_ranges.py
"$SOCIAL_RL_STORE/envs/data/bin/python" scripts/finalize_model.py
