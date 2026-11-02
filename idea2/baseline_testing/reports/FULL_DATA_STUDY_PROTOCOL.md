# Full external-data SFT and full EgoNormia evaluation

The user explicitly requested expansion beyond the 400-ID panel and training
on the full external datasets, then explicitly required no EgoNormia overlap
in training. This protocol freezes the model list and fixed training recipe
before opening the former final partition. It supersedes the earlier
400-ID-only evaluation boundary. Full EgoNormia results will contain both
previous development examples and the formerly reserved groups; future
iterations evaluated on this release must acknowledge that it is exposed.

## Actual prior coverage

R³-FDT has 41,529 annotated questions over 2,812 videos. Prior training used
1,262 questions from 84 accessible training videos, sometimes repeatedly.
PsycheMME currently releases 3,638 SFT questions. The overlap audit removed
104 questions from 20 Ego4D source videos shared with any EgoNormia
partition, leaving 3,534 eligible questions. The earlier choice-only study
trained on 3,032 questions, retaining 389 validation questions and excluding
111 open questions plus one choice with option-letter references.

## Strict training boundary

`social_rl.leakage.check_training` runs before every full-data training session.
It checks a SHA256-frozen ID-only list of all source-video groups in the
actual official 1,853-example annotation file. This checkout indexes 1,075
unique source identifiers (the published 1,077 figure is not substituted).
PsycheMME questions must be in the audited eligible ID list and their source
video must be absent from the full EgoNormia source list. All clips from
any shared source video are excluded, irrespective of timestamp or label.
R³ source IDs must also be absent. Training manifests must contain only
external training records, with unique IDs. Unknown/disallowed examples
fail closed. Tests confirmed the eligible manifests pass and an unknown
PsycheMME ID is rejected. No EgoNormia question, answer, video, or rationale
is a training example. Source-ID checks cannot certify undocumented
cross-dataset alternate uploads or unknown original-model pretraining data.

Full PsycheMME SFT now includes all 3,534 eligible examples, including open
answers and previously held-out source examples. Option references are
remapped alongside shuffled choices, so the formerly excluded choice can
be retained. Eight of the released 32 frames are used. Original observation
and answer targets are retained; supplied transcripts/history remain context.
No fabricated norm labels or generated reasoning targets are added.

Full R³ media acquisition attempts every one of the 2,812 annotated videos
through the upstream source links. Acquisition logs and full coverage reports
are retained. Only videos validated as decodable enter the final training
manifest. Inaccessible footage is a coverage blocker, not silently replaced
with text. The names `full_r3` denote the intended complete acquisition;
reported training claims must state the actual usable records/videos. Source
validation/test partitions become training in this phase; their subsequent
scores cannot be called held-out source performance.

## Frozen models and recipe

The full registry is `reports/results/full_study_registry.json`; its hash is
recorded in `full_study_boundary.json`. For each original model family,
Qwen3.5-9B and Qwen3.8-27B, train three fresh paths:

1. Original model → all usable R³-FDT.
2. Original model → all eligible PsycheMME.
3. Full R³-SFT → all eligible PsycheMME.

Each stage makes one fixed pass through its manifest, seed 777, eight video
frames, rank-8 language LoRA, NF4/FP16, AdamW learning rate 5e-6 and zero
weight decay. Targets are answer-only for R³ and supplied observation plus
answer for PsycheMME; open PsycheMME answers are supported. A 512-token
assistant-target ceiling fails rather than silently truncates. Nonfinite
gradients trigger a logged retry of the same example, with a bounded failure
limit. Optimizer/scaler/RNG state is checkpointed every 250 updates, keeping
the most recent two checkpoints. A training job resubmits a continuation
if its 3.2-hour work budget expires. On completion it schedules evaluation
and, for R³, the downstream PsycheMME stage. There is no selection using full
EgoNormia labels. These full-data recipes are fixed before results.

The registry additionally evaluates original 9B/27B baselines and saved
9B SFT-100, both 300-update PsycheMME variants, RLOO, GRPO, the four
source-selected epoch variants, and the extra R³-SFT control. Historical
models retain their actual pilot-data provenance; evaluation on a larger
benchmark does not turn them into full-data training runs. NormLens/MUSON
continuations were omitted because they did not improve transfer.

## Full benchmark evaluation

Use all 1,853 official repository examples, video_prev only, 1,024 output
tokens, ordinary native action/justification/sensibility prompts and greedy
non-thinking generation. Pin original model releases and retain explicit
model-path/release metadata. The full evaluator requires the frozen
manifest hash via `--full-evaluation-protocol`. The training path has no
benchmark-label interface.

Attempt download/validation of every preceding clip. Missing media receive
explicit failure records and remain in the full denominator; report accessible
coverage as well as the full attempted score. No missing example is silently
dropped. Shards contain 100 items (last shard 53) and resume automatically
when their time budget is exhausted. A model is summarized only after all
1,853 unique IDs are present. Report native action, justification, joint,
sensibility IoU, verified membership, category results, and failure counts.
Additionally report the previous 400 IDs, the 1,025 formerly reserved examples
from different source groups, and the remaining 428 development examples
separately. These cohorts are experiment-specific, not official train/test
splits. Use paired source-video-cluster intervals for model comparisons.

The new source-group cohort provides a first transfer check for the frozen
models. The full aggregate mixes development and new examples. Additional
model/recipe changes made after viewing these results must be labeled
exploratory and cannot claim an untouched benchmark test.

## Execution and artifacts

Jobs are recorded in `reports/results/full_study_jobs.json`. Media arrays
50115735 (R³) and 50115736 (EgoNormia) cover 16 shards each. The initial
9B/27B PsycheMME jobs are 50115936/50115937. R³ training dispatch follows
complete acquisition; the complete train/evaluate chain resubmits itself
through Slurm and records child job IDs in logs. Long training and inference
remain on one V100-32 GB per job. All media and checkpoints stay on Ocean.
Code, frozen registry, protocol, and measurements are synced to the team
repository under `idea2/baseline_testing`.

The initially submitted PsycheMME training jobs were stopped during model
loading and replaced by 50115936/50115937 so that the mandatory runtime
source-exclusion guard, explicit zero weight decay, and media-acquisition
evaluation dependencies apply from their first update.

Technical amendment during monitoring: see
`PSYCHEMME_FRAME_REPRESENTATION_CORRECTION.md`. A source-frame audit found
that the legacy frame path invented timestamps. Distinct corrected runs
use the original 32 ordered images after V100 smoke tests. Original
weights/results/registry remain archived; pending R³→PsycheMME stages use
the explicitly amended specs. The correction is not a new untouched test.
