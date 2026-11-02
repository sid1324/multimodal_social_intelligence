#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --job-name=q38-base
#SBATCH --output=logs/q38-base-%A_%a.log
#SBATCH --error=logs/q38-base-%A_%a.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/qwen38/bin/python scripts/evaluate.py --model-path "$SOCIAL_RL_STORE/models/Qwen3.8-27B" --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" --output "$SOCIAL_RL_STORE/runs/qwen38_base_ego1024_part${SLURM_ARRAY_TASK_ID}" --offset "$((SLURM_ARRAY_TASK_ID*100))" --limit 100 --conditions video --max-new-tokens 1024 --max-hours 3.5
