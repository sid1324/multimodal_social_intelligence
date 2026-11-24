#!/bin/bash
# Login-node loop (Nov 24 to 25, approved by Yash): keep each interactive lane busy with a resumable driver job while work is runnable.
# submit.py enforces the queue rules and the 150 SU pause. Ends on STOP.md, no work left, or the end time (epoch seconds in $1).
cd /work/hdd/bjuv/ylothe/multimodal_social_intelligence/idea3_face_bias/final_omni
source <(grep -E '^(W=|export)' slurm/env.sh); END=$1
while [ $(date +%s) -lt $END ]; do
  [ -f results/STOP.md ] && { echo "STOP.md"; break; }
  line=$(OMP_NUM_THREADS=4 ACTS_DIR=/tmp/i3env_ylothe/acts /tmp/i3env_ylothe/env/bin/python src/driver.py --check 2>/dev/null | grep RUNNABLE)
  n=$(echo "$line" | awk '{print $2}'); echo "$(date +%H:%M) $line"
  [ "$n" = "STOP" ] && break
  echo "$line" | grep -q "LEFT 0" && { echo "ALL DONE"; break; }
  if [ "${n:-0}" -gt 0 ]; then
    for spec in "gpuA40x4-interactive 2 0.95" "gpuA100x4-interactive 1 0.95" "gpuH200x8-interactive 1 0.417"; do
      set -- $spec; p=$1; np=$2; h=$3
      if [ -z "$(squeue -u ylothe -h -p $p -o %i)" ]; then python3 slurm/submit.py drv "python src/driver.py" --part $p --procs $np --hours $h; sleep 5; fi
    done
  fi
  sleep 60
done
echo "keepalive2 ended $(date)"
