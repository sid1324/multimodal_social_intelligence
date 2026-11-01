#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=4
#SBATCH --mem=7600M
#SBATCH --time=04:00:00
#SBATCH --job-name=q38-prepare
#SBATCH --output=logs/q38-prepare-%j.log
#SBATCH --error=logs/q38-prepare-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ROOT=/ocean/projects/cis260338p/apatawar/social_norm_rl
"$ROOT/envs/data/bin/python" scripts/qwen38_preflight.py
"$ROOT/envs/qwen35/bin/python" -m venv "$ROOT/envs/qwen38"
printf '%s\n' "$ROOT/envs/qwen35/lib/python3.12/site-packages" > "$ROOT/envs/qwen38/lib/python3.12/site-packages/qwen35_dependencies.pth"
"$ROOT/envs/qwen38/bin/pip" install --ignore-installed transformers==5.22.1
"$ROOT/envs/qwen38/bin/python" -c 'import transformers; print(transformers.__version__)'
