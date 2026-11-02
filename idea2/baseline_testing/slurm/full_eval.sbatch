#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=ego-full
#SBATCH --output=logs/ego-full-%A_%a.log
#SBATCH --error=logs/ego-full-%A_%a.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
SPEC="$1"
ADAPTER="${2:-none}"
PYTHON=$(python -c 'import json,sys;s=json.load(open(sys.argv[1]));print("/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/qwen38/bin/python" if "3.8" in s.get("model_path","") else sys.executable)' "$SPEC")
"$PYTHON" scripts/full_eval_shard.py --spec "$SPEC" --adapter "$ADAPTER" --shard "$SLURM_ARRAY_TASK_ID"
