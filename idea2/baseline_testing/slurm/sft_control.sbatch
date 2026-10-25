#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-sft-control
#SBATCH --output=logs/sft-control-%j.log
#SBATCH --error=logs/sft-control-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/train.py --manifest "$SOCIAL_RL_STORE/data/manifests/r3_train_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/sft" --method sft --steps 100 --max-hours 1.4
ADAPTER_PATH=$(python -c 'import json; from social_rl.common import STORE; print(json.load(open(STORE/"runs/sft/latest.json"))["checkpoint"])')
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_matched.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/sft_ego" --adapter "$ADAPTER_PATH" \
  --conditions video --max-hours 1.2
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/r3_val_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/sft_r3_val" --adapter "$ADAPTER_PATH" \
  --conditions video --limit 40 --max-hours 0.25
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/r3_test_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/sft_r3_test" --adapter "$ADAPTER_PATH" \
  --conditions video --limit 60 --max-hours 0.3
python scripts/summarize.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego" "$SOCIAL_RL_STORE/runs/rloo_ego" \
  "$SOCIAL_RL_STORE/runs/sft_ego" --output reports/generated/with_sft_control
