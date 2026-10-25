#!/bin/bash
#SBATCH -A cis260338p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out
# One follow-up arm: train (seed $SEED) on data/train_$ARM.jsonl, then evaluate on SAT static held-out + EgoNormia.
# usage: sbatch -J idea5-<name> --export=ALL,ARM=spatial,SEED=2,NAME=spatial_s2 run_arm.sbatch
set -euo pipefail
cd $PROJECT/idea5_pilot
module load anaconda3/2024.10-1
source venv/bin/activate
export HF_HOME=$PROJECT/idea5_pilot/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
nvidia-smi -L
if [ "$ARM" = color ] && [ ! -f data/train_color.jsonl ]; then
  HF_HUB_OFFLINE=0 python prep_narrow.py
fi
export HF_HUB_OFFLINE=1
[ -f adapters/$NAME/adapter_config.json ] || python train_lora.py data/train_$ARM.jsonl adapters/$NAME --seed $SEED
[ -f results/$NAME.json ] || python eval_arm.py adapters/$NAME results/$NAME.json
echo ARM_DONE $NAME
