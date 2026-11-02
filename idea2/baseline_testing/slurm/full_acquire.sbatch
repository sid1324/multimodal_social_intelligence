#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=4
#SBATCH --mem=7600M
#SBATCH --time=06:00:00
#SBATCH --array=0-15%4
#SBATCH --job-name=full-media
#SBATCH --output=logs/full-media-%A_%a.log
#SBATCH --error=logs/full-media-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/data/bin/python scripts/full_data_prepare.py --task "$1" --shard "$SLURM_ARRAY_TASK_ID"
