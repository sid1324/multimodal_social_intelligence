#!/bin/bash
#SBATCH -J idea5-static
#SBATCH -A cis260338p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# Manipulation check: held-out SAT static accuracy for base / spatial / control.
set -euo pipefail
cd $PROJECT/idea5_pilot
module load anaconda3/2024.10-1
source venv/bin/activate
export HF_HOME=$PROJECT/idea5_pilot/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python eval_static.py
