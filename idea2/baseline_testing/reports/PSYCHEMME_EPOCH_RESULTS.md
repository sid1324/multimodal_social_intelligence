# Epoch SFT replication and 27B feasibility: measured outcomes

All four 3,032-update SFT jobs, checkpoint-selection evaluations, R³
retention evaluations, the R³ continuation control, and the 27B smoke test
completed successfully. This is exploratory development on the same
400-ID EgoNormia panel, with a 1,024-token cap. The final partition remains
reserved. The protocol is `PSYCHEMME_EPOCH_AND_QWEN38_PROTOCOL.md`.

| Training condition | Seed | Source-selected update | PsycheMME val120 | Ego action | Ego justification | Ego joint | Sensibility IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base → PsycheMME, epoch study | 777 | 3032 | 80.0% | 44.0% | 43.0% | 35.8% | 46.5% |
| Base → PsycheMME, epoch study | 1777 | 2250 | 79.2% | 49.8% | 47.8% | **41.2%** | 47.3% |
| R³-SFT → PsycheMME, epoch study | 777 | 3032 | 80.0% | 41.5% | 44.8% | 35.2% | 46.2% |
| R³-SFT → PsycheMME, epoch study | 1777 | 3032 | 78.3% | 42.0% | 43.5% | 35.8% | 45.0% |
| R³-SFT → extra R³-SFT control | 777 | 3032 fixed | — | 49.0% | 44.2% | 38.0% | 49.2% |
| Previous R³-SFT → PsycheMME pilot | 777 | 300 fixed | 77.5% | 50.7% | 45.2% | 38.8% | 49.2% |
| Previous R³-SFT control | 777 | 100 fixed | — | 50.0% | 41.8% | 35.2% | 48.7% |

The exact best score is 165/400 = 41.25%; table values follow the project's
one-decimal rounding. Checkpoints were selected solely by highest source
validation accuracy among 750/1500/2250/3032, ties to earlier steps. All
training runs finished 3,032 accepted updates; the seed-1777 base run's
selected checkpoint is earlier. There are no inference errors in the main
400-item evaluations. Invalid-example counts are 1/1/1/2 for the four
epoch conditions in table order, versus 9 for the R³ continuation control.

The best epoch score is +2.5 points versus the previous 38.75% pilot, with
source-video-cluster paired 95% bootstrap interval −0.8 to +5.8 points.
The other epoch differences versus that pilot are −3.0 (−7.1 to +0.8),
−3.5 (−8.0 to +0.8), and −3.0 (−7.6 to +1.5). The R³ continuation control
differs by −0.75 (−3.5 to +2.0). These sampling intervals do not measure
seed uncertainty and are not adjusted for choosing the best of four runs.

The two-seed mean is 38.5% joint for the base starts and 35.5% for the
R³-SFT starts. Two seeds provide limited replication. The 5.5-point spread
between base-start seeds is substantial. Higher source validation accuracy
does not consistently predict transfer: the 80%-validation models score
35.75% and 35.25% on EgoNormia, while the 79.17%-validation checkpoint scores
41.25%. More source training is therefore not a reliable transfer improvement
in this study. Sensibility IoU decreases across all four epoch conditions
relative to the pilot, despite fewer invalid outputs. A single best seed
should not be presented as a stable improvement.

On the 79 accessible R³ retention questions, the selected base starts score
52/79 (65.8%) for seed 777 and 56/79 (70.9%) for seed 1777; the R³-SFT starts
score 70/79 (88.6%) and 71/79 (89.9%). These 79-question figures are not
directly comparable to older 60-question source results. The control matches
additional optimizer updates and learning rate, not GPU hours or target-token
counts. The base→PsycheMME conditions avoid the initial R³ training exposure.

## Analysis integrity and incomplete controls

The original automated summary silently merged four runs and the R³ control
because each prediction directory was named `ego1024_video400`. Raw
predictions were intact. `summarize.py` now uses parent-qualified labels when
basenames repeat, rejects duplicate labels and duplicate per-condition IDs,
and the aggregate was recomputed. This affected the new epoch aggregate,
not the earlier results whose directory basenames were unique.

Although all evaluation jobs exited successfully, their one-hour visual-control
budgets produced only 163/200, 118/200, 127/200 and 135/200 records (base777,
base1777, R³777, R³1777). The original-model controls completed 200/200.
The four partial controls have been resumed in separate jobs, listed in
`results/next_experiment_jobs.json`. Their results should be compared only
at matched coverage. The main EgoNormia table above is complete at 400/400
for every condition.

## Qwen3.8-27B

Pinned Qwen3.8-27B NF4 on one V100-32 GB passed video inference and a short
language-LoRA backward pass with finite gradients. Peak reserved memory was
21.40 GB during inference and 24.01 GB through backward. One 55-token
response took 29.92 seconds including prefill, excluding preprocessing and
model loading; this single cold example is not a stable throughput benchmark.
This does not test optimizer state allocation, a full training run, longer
visual contexts, or reasoning with extended generation.

The unmodified 27B EgoNormia baseline is now queued as four 100-ID shards
at a 1,024-token cap, array `50115599` (concurrency limit two, one V100 per
task). Each shard carries explicit model path and pinned-release metadata.
Check coverage before aggregating and resume any timed-out shard. No 27B
EgoNormia score or SFT outcome is available yet.

Machine-readable epoch curves, paired summaries, and 27B smoke measurements
are in `reports/results/psychemme_epoch_*` and
`reports/results/qwen38_v100_smoke.json`. Raw predictions and adapters are
on Ocean. These results motivate source-alignment/seed controls and a stronger
unmodified baseline before further RL optimization.
