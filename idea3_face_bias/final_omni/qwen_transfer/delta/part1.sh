#!/bin/bash
# Qwen3-VL transfer, Part 1 on Delta (Yash asked Nov 25 21:55 CT, as a backup to Colab): repro, perc, base, gate.
# Same code as the Colab notebook. GPU env: the Nemotron env (torch 2.9.1+cu128, pillow 12.3.0) with transformers 4.57.6 and
# huggingface_hub 0.36.2 overlaid (copied from the venv that passed the CPU smoke test). Resumable; stops cleanly 5 min before the limit.
set +e
W=/work/hdd/bjuv/ylothe/idea3_final
export HF_HOME=$W/hfq_qwen HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$W/qt_overlay
export P777=$W DATA_DIR=$W/data/egonormia REF_DIR=$W/data/ref PYTHONUNBUFFERED=1
export STOP_AT=$(( $(date +%s) + ${QT_MINUTES:-52} * 60 ))
cd "$(dirname "$0")/.."
python -c "import torch, transformers, huggingface_hub, PIL; print('versions', torch.__version__, transformers.__version__, huggingface_hub.__version__, PIL.__version__, torch.cuda.get_device_name(0))"
[ -f results/gate.json ] && { echo "Part 1 already done"; exit 0; }
for s in repro perc base gate; do
  echo "=== $s $(date +%H:%M)"; python src/transfer.py $s; rc=$?
  echo "=== $s exit $rc $(date +%H:%M)"
  [ -f results/STOP.md ] && { echo "STOPPED BY A GATE"; cat results/STOP.md; exit 0; }
  [ $rc -eq 75 ] && { echo "time limit reached; resubmit to continue"; exit 0; }
  [ $rc -ne 0 ] && { echo "stage failed"; exit $rc; }
done
echo "PART 1 DONE"
