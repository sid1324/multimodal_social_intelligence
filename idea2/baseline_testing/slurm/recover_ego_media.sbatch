#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=06:00:00
#SBATCH --job-name=ego-media-recover
#SBATCH --output=logs/ego-media-recover-%j.log
#SBATCH --error=logs/ego-media-recover-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/data/bin/python scripts/recover_ego_media.py
