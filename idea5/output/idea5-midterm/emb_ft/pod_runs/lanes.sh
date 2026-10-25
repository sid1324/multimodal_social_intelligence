#!/bin/bash
# Three GPU lanes on the A100: zero-shot + C2 seeds, C5 seeds, X1 seeds.
cd /root/emb_ft
mkdir -p logs
run() { python emb_ft.py "$@" 2>&1 | grep -v -i -E "warn|deprecat|detach" ; }
( run --zeroshot --conds C2,C5,X1 --out preds/zeroshot > logs/zeroshot.log
  for s in 0 1; do run --cond C2 --seed $s --out runs/C2_s$s > logs/C2_s$s.log; done ) &
( for s in 0 1; do run --cond C5 --seed $s --out runs/C5_s$s > logs/C5_s$s.log; done ) &
( for s in 0 1; do run --cond X1 --seed $s --out runs/X1_s$s > logs/X1_s$s.log; done ) &
wait
echo ALL_DONE > logs/ALL_DONE
