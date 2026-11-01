#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=01:00:00
#SBATCH --job-name=social-muson-prepare
#SBATCH --output=logs/muson-prepare-%j.log
#SBATCH --error=logs/muson-prepare-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$SOCIAL_RL_STORE/envs/data/bin/python" scripts/prepare_muson.py
