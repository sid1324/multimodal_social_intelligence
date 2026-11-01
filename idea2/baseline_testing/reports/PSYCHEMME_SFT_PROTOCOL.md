# PsycheMME SFT transfer experiment (pre-result protocol)

This extends the original R³-FDT reasoning study with independent normative
social-video supervision. PsycheMME is a training source; EgoNormia remains
the target evaluation benchmark. The baseline comparison is exploratory on
the pre-existing 400-ID panel. The final EgoNormia partition is reserved.

## Data and leakage

Pin PsycheMME training revision `3b80a47473802b160b54e760d5dca070efd96f4c`.
See `PSYCHEMME_AUDIT_AND_PLAN.md` and `results/psychemme_audit.json`.
Remove all 104 current SFT questions whose Ego4D source-video IDs occur in
either EgoNormia partition. Exclude open questions for the first
multiple-choice experiment. Exclude one question whose wording refers to a
specific option letter; the letter would change after shuffling. Split
remaining choices using SHA256 of `(source, source_video_id)` with 10% held
out for validation. EgoCom camera views additionally group by day and
conversation. Result: 3,032 train questions in 254 groups and 389
validation questions in 38 groups. The groups are disjoint. This source
validation is for diagnostics; EgoNormia is not used to tune training.

Use eight frames from each released 32-frame set, at indices
`[0,4,9,13,18,22,27,31]`, preserving temporal order. Convert them to four
two-frame video segments through the existing Qwen3.5 native-video path,
matching the EgoNormia representation as closely as the released static
frame archive permits. Released frames are at most 256 px. The archive SHA256
is in the split audit. The source prompt includes provided ASR where present.
The target is the one supplied observation plus the remapped correct answer.
Shuffle options deterministically per example/visit with seed 777. Loss is
mean assistant-target token cross entropy. No EgoNormia data enter training.

## Conditions and compute

Train Qwen3.5-9B NF4/FP16 with frozen vision model and rank-8 language LoRA
on one V100-32 GB per Slurm job. The common training set, seed, learning
rate `5e-6`, 300 accepted updates, 128-token target ceiling, frame input,
optimizer, and prompt format apply to both conditions:

1. `base`: initialize a new LoRA adapter on the original model.
2. `r3sft`: continue the saved 100-update R³-FDT answer-only SFT adapter.

The first condition estimates PsycheMME's independent contribution. The
second tests complementarity with the existing R³ pathway. These are not
training-compute-matched to the prior R³ control because of different task
structure and target lengths. Report that limitation.

Run a three-step base smoke test first. The first smoke job `50103888`
failed on nonfinite gradients at the first step, before a model update.
The implementation now starts the FP16 gradient scaler at 1024 and logs
and skips any nonfinite-gradient attempt; accepted optimizer updates remain
the step count, with a three-attempts-per-step stop bound. The replacement
smoke job `50104051` completed successfully: eight attempts, five logged
overflow skips, three accepted updates in 48 seconds after model loading,
and 12.85 GB peak GPU memory reserved. Main jobs depend on its success.
Checkpoint every 50 updates and save optimizer/scaler/RNG state. If a job
stops before 300 steps, resume explicitly and record actual steps. Do not
accept dependent evaluations of a missing 300-step checkpoint.

## Evaluation

Use the exact user-supplied 400 EgoNormia IDs and preceding videos, official
action/justification/sensibility prompts, 1,024 maximum new tokens, and all
attempted examples in the denominator. Record action accuracy,
justification accuracy, joint accuracy, sensibility IoU, invalid counts,
and source-video-cluster paired intervals against both original model and
R³-SFT. Two separate jobs evaluate the two adapters; a CPU analysis runs
only after both finish. Independently score the first 120 sorted source
validation IDs for answer accuracy and report that source panel coverage.
No result is assumed by this protocol. The reserved final EgoNormia
partition remains unopened through exploratory model development.

## Submitted Slurm chain

- Three-step replacement smoke: `50104051`.
- Training: base `50104052`; R³-SFT continuation `50104053`.
- EgoNormia evaluations: base `50104080`; R³-SFT continuation `50104081`.
- PsycheMME source validation: base `50104082`; R³-SFT continuation `50104083`.
- Paired EgoNormia analysis: `50104084` after both EgoNormia evaluations.

Raw media, model artifacts, and predictions remain under
`/ocean/projects/cis260338p/apatawar/social_norm_rl/`.
