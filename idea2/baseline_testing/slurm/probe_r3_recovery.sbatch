#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=r3-cooldown-probe
#SBATCH --output=logs/r3-cooldown-probe-%j.log
#SBATCH --error=logs/r3-cooldown-probe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/data/bin/python scripts/probe_r3_recovery.py
