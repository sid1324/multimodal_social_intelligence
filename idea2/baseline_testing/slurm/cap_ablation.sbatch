#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=social-cap-ablation
#SBATCH --output=logs/cap-ablation-%j.log
#SBATCH --error=logs/cap-ablation-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/base_ego_cap384" --conditions video --limit 40 \
  --max-new-tokens 384 --max-hours 1.5
python scripts/summarize.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego" "$SOCIAL_RL_STORE/runs/base_ego_cap384" \
  --output reports/generated/cap_ablation
