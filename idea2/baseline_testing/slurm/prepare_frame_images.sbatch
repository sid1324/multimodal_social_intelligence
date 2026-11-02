#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=01:00:00
#SBATCH --job-name=psy-images-prepare
#SBATCH --output=logs/psy-images-prepare-%j.log
#SBATCH --error=logs/psy-images-prepare-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/data/bin/python scripts/prepare_psy_frame_images.py
