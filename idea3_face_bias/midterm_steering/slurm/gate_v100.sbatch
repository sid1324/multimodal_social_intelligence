#!/bin/bash
#SBATCH -J idea3_gate_v100
#SBATCH -A cis260338p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x_%j.out
# First GPU job: correctness check, then the gate on train items with residual capture.
set -euo pipefail
umask 002
export P777=/ocean/projects/cis260338p/ylothe
export HF_HOME=/ocean/projects/cis260338p/shared/hf_cache HF_HUB_OFFLINE=1
export CACHE_DIR=$P777/cache OUT_DIR=$P777/outputs
module load cuda/12.4.0
source $P777/.venv/bin/activate
cd $P777/multimodal_social_intelligence/idea3_face_bias/midterm_steering
python -c "import json; r=json.load(open('$P777/data/egonormia/verify_report.json')); assert r['raw_sha_ok']==1853, r['raw_sha_ok']; print('images verified')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python src/run_gate.py --n_gate 300 --n_check 20
