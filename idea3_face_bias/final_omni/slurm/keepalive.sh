#!/bin/bash
# Login-node loop: whenever an interactive partition is free and unlocked work exists, submit one 25-min driver job there
# (slurm/submit.py applies the queue rules and the SU cap). Ends on STOP.md, no work left, or the given time (HHMM CT).
cd /work/hdd/bjuv/ylothe/multimodal_social_intelligence/idea3_face_bias/final_omni
source <(grep -E '^(W=|export)' slurm/env.sh); END=${1:-0655}
while [ $(date +%H%M) -lt $END ]; do
  [ -f results/STOP.md ] && { echo "STOP.md"; break; }
  line=$(OMP_NUM_THREADS=4 /tmp/i3env_ylothe/env/bin/python src/driver.py --check 2>/dev/null | grep RUNNABLE)
  n=$(echo "$line" | awk '{print $2}')
  echo "$(date +%H:%M) $line"
  if [ "$n" = "STOP" ]; then break; fi
  if echo "$line" | grep -q "LEFT 0"; then echo "ALL DONE"; break; fi
  for p in gpuA40x4-interactive gpuA100x4-interactive; do
    pend=$(squeue -u ylothe -h -t PD -o %i | wc -l)
    if [ "${n:-0}" -gt "$pend" ] && [ -z "$(squeue -u ylothe -h -p $p -o %i)" ]; then python3 slurm/submit.py drv "python src/driver.py" --part $p; sleep 5; fi
  done
  sleep 45
done
