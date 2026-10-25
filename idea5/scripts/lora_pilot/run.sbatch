#!/bin/bash
#SBATCH -J idea5-pilot
#SBATCH -A cis260338p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Spatial-vs-control LoRA pilot. Every stage is skipped if its output exists, so a resubmit resumes.
set -euo pipefail
cd $PROJECT/idea5_pilot
module load anaconda3/2024.10-1
source venv/bin/activate
export HF_HOME=$PROJECT/idea5_pilot/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi -L
# Model and data downloads run here: compute nodes reach Hugging Face ~40x faster than the login node.
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
[ -f hf/READY ] || { hf download Qwen/Qwen3-VL-8B-Instruct > logs/download.log 2>&1 && touch hf/READY; }
[ -f data/READY ] || { python prep_data.py > logs/prep.log 2>&1 && touch data/READY; }
tail -8 logs/prep.log
export HF_HUB_OFFLINE=1
[ -f results/smoke_eval.json ] || python eval_models.py base results/smoke_eval.json --limit 4
[ -f adapters/smoke/adapter_config.json ] || python train_lora.py data/train_spatial.jsonl adapters/smoke --max-steps 16
for arm in spatial control; do
  [ -f adapters/$arm/adapter_config.json ] || python train_lora.py data/train_$arm.jsonl adapters/$arm
done
for m in base adapters/spatial adapters/control; do
  name=$(basename $m)
  [ -f results/$name.json ] || python eval_models.py $m results/$name.json
done
echo PILOT_DONE
