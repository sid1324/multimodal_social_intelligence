# Midterm protocol — frozen before model results

Date: 2026-10-23. Deadline: approximately two days. One V100-32 GB per job.

## Research claim

Does correctness-reward RL on R3-FDT improve video-grounded social reasoning
and transfer to normative action/justification decisions in EgoNormia?
R3-FDT descriptive causal QA does not provide normative supervision.
Answer correctness does not verify generated explanations.

## Model and inputs

Qwen/Qwen3.5-9B at the revision in configs/midterm.json. NF4 double
quantization with FP16 computation, frozen vision tower, language-only LoRA.
Eight uniformly sampled frames, aspect ratio retained, explicit resolution cap.
For Transformers 5.3.0/Qwen3.5 compatibility, feed those frames as four
ordered two-frame video segments with original frame timestamps. A single
eight-frame processor call produced four video-token runs but one grid row,
causing the initial V100 smoke test to fail before inference.
Native video processor with timestamps; no transcripts or human descriptions.
Dialogue-dependent questions are retained in the initial R3 coverage and
reported as a limitation; no post-hoc filtering based on model correctness.
Any smoke-test-driven resource changes must be recorded before scoring.

## Data boundary

R3-FDT only supplies optimization and checkpoint selection. Source-movie
grouping where upstream metadata permits, video grouping otherwise. Stable
hash partition into 80% training, 10% validation, 10% test groups. Accessible
subsets are selected by stable hash, never by answer or baseline correctness.

EgoNormia is never training data. The team supplied the exact 400 IDs in
`egonormia_400_sample_ids.txt` before any model baseline run. That file fixes
the midterm panel in its given order. All 400 occur in official repository
annotations; 376 occur in the HF parquet and 24 use repository annotations.
Media access is checked separately and coverage must accompany every result.
If the evaluation time budget stops before all 400 items, the attempted IDs
are a prefix of the team's supplied order. Such results describe that prefix,
not an unbiased estimate for all 400 or the full benchmark.
After the initial matched 140-ID three-condition run, complete a separate
correct-video-only analysis on the remaining 260 supplied IDs for base, RLOO,
and SFT. Merge only identical 192-token video predictions, preserving exact
ID order and prediction hashes. The 400-ID correct-video comparison then has
full panel coverage; no-video and mismatched-video diagnostics remain limited
to the original 140 matched IDs. This extension still uses no final holdout
source-video group or EgoNormia label for training.
All examples sharing a source-video group with the panel, plus already
inspected public examples, are assigned to the midterm partition; all other
source groups are reserved for final evaluation. This is an experiment-specific
split, not an official split. No final-partition example-level inspection,
inference, or scoring during midterm. Verified membership is reported but does
not change panel selection.

## Initial comparisons

1. Original NF4 checkpoint, correct video.
2. Same checkpoint, no video.
3. Same checkpoint, mismatched video from a different source group.
4. RL adapter on the same NF4 checkpoint, correct video.
5. RL adapter with no video and mismatched video if runtime allows.
6. Supervised adapter on identical accessible R3 training data if time allows.

The supervised control is now queued after the main RL comparison and cap
diagnostic. It uses the same frozen model, LoRA targets, optimizer settings,
R3 training manifest, and nominal 100 optimizer updates. Its answer-only
cross-entropy objective differs from RLOO and may visit different training
examples because RLOO skips uniform-reward groups. This is a useful control
for generic task adaptation, not a compute-matched algorithm comparison.

A separate diagnostic response-cap ablation is planned on the first 40 panel
IDs, selected by the frozen file order rather than correctness. It repeats
the original model's correct-video prompt with a 384-token cap, versus the
192-token initial setting, to distinguish incomplete answers from semantic
errors. It is not part of the primary fixed-cap RL comparison.
Because that exploratory ablation showed a large formatting effect, a matching
384-token RL-adapter run on the same 40 IDs is queued after the SFT control.
Together these form a small model-by-cap diagnostic, still separate from the
140-ID primary comparison.

All conditions use the same evaluation panel and generation settings. An RL
checkpoint is selected using R3 validation only; a fixed final step is used if
no validation checkpoint selection can be completed. One seed is an initial
pilot, not replicated evidence. A small pilot need not improve performance.

## RL formulation

Use RLOO with four on-policy completions per video/question. Correct, valid
answers receive 1, all others 0. A bounded explanation is permitted but not
separately rewarded. Remap labels after random option permutations. The
advantage is reward minus the mean reward of the other three completions.
Use one update per group, no reward standard-deviation scaling, a fixed
completion-cap denominator, and a sampled reference KL penalty (beta=0.02).
Sampling uses temperature=1, top_p=1, top_k=0 to match training probabilities.
Zero-variation groups are logged and skipped; this selects for examples with
within-group reward variation and must be acknowledged in analysis. Record
actual loss, rollout truncations, optimizer steps, and effective coverage.
The change from the initially suggested GRPO to this simple on-policy RLOO
pilot avoids a critic, replay buffer, or separate generation server and makes
the initial implementation inspectable within the two-day deadline.

## EgoNormia protocol

Use video_prev.mp4 only: video_during.mp4 can reveal future behavior and is
excluded. Preserve official action, selected-action-conditioned justification,
and sensibility prompts. No ground-truth action is passed to the justification
prompt. Preserve release option order for the native-protocol comparison.
Do not load prior models' evaluation results.

Primary: joint action-and-justification accuracy. Secondary: action accuracy,
justification accuracy, mean per-example sensibility IoU. Count all attempted
examples, including inference errors and invalid outputs, in the primary
denominator. The native-style last-integer parser can assign a choice even
when a response fails our stricter final-line validity check; validity and
token-cap rates are reported separately. Also export the official scorer
format and document differences from upstream failure dropping or permissive
parsing. Categories use the
correct action's taxonomy, with overlapping membership reported. Bootstrap
by source-video group. Coverage is always displayed.

## Error analysis and report

Audit 20–30 errors and several successes from the midterm panel. Inspect the
sampled evidence and full allowed preceding clip. Classify perception,
temporal/causal reasoning, participant-role inference, descriptive/normative
confusion, norm conflict, action/justification mismatch, and output-format
failures. These are analyst hypotheses, not inferred hidden reasoning.
Use paired no-video/mismatched-video results as diagnostics; disruption alone
does not prove grounding. The no-video control also replaces the standard
introductory phrase with a blind-context instruction, so its effect is not a
pure vision ablation. Future methods must be motivated by actual errors.

## Access and licenses

R3 annotations: CC BY-NC 4.0; CondensedMovies videos retain original rights.
EgoNormia: use official release and preserve its attribution/license notices.
No video redistribution. Record failed media acquisition and distribution
changes from incomplete access. Missing video is never silently replaced by
text or synthetic clips in a reported training experiment.
