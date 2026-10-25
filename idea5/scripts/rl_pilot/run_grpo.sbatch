#!/bin/bash
#SBATCH -A cis260338p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# One GRPO arm then its test-set eval. usage:
#   sbatch -J rl-R1_s0 --export=ALL,ARM=R1,SEED=0,STEPS=300,NAME=R1_s0 run_grpo.sbatch
#   SMOKE=1 adds --limit 16 to the eval and writes under smoke_*.
set -euo pipefail
cd $PROJECT/rl_pilot
module load anaconda3/2024.10-1
source $PROJECT/idea5_pilot/venv/bin/activate
export HF_HOME=$PROJECT/idea5_pilot/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
# STEPS may come from steps.txt (set after the smoke test) so queued jobs pick up the final value.
[ -z "${STEPS:-}" ] && STEPS=$(cat steps.txt)
echo "arm $ARM seed $SEED steps $STEPS"
FMT=$(python -c "import rl_common as rc; print(rc.ARMS['$ARM'])")
[ -f adapters/$NAME/state.json ] && grep -q "\"step\": $STEPS" adapters/$NAME/state.json || python grpo_train.py --arm $ARM --out adapters/$NAME --steps $STEPS --seed $SEED ${EXTRA:-}
LIMIT=${SMOKE:+--limit 16}
[ -f results/$NAME.json ] || python eval_rl.py --format $FMT --adapter adapters/$NAME --out results/$NAME.json $LIMIT
echo ARM_DONE $NAME
