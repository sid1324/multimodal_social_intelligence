# PsycheMME training audit and proposed SFT study

Audit date: 2026-11-01. This is a dataset assessment; no PsycheMME model has
been trained or evaluated by this project yet.

## Assessment

PsycheMME is a promising secondary training source for our transfer question.
Its training questions include embodied social actions, intervention timing,
and boundaries, alongside state and need inference. These targets are closer
to EgoNormia's normative decisions than R3-FDT's descriptive movie QA.
This is a hypothesis about transfer, not a measured improvement.

The official paper reports Qwen3.5-4B PsycheMME Raw/Verified scores of
47.68/37.71 before adaptation, 72.58/52.13 after SFT, and 73.19/52.62 after
SFT+GRPO. These are their own benchmark scores, not EgoNormia scores or
evidence for the same gain on our stronger 9B model. The paper's SFT uses
answers plus one annotated observation, 32 frames, available ASR/history,
one epoch, and language LoRA. Its GRPO reward checks answer correctness and
output validity; rationale verification is an evaluation measure, not the
GRPO training reward.

Sources:

- https://arxiv.org/html/2610.28625v1 (Sections 3.6, 4.3, Appendix E)
- https://github.com/psychemme/PsycheMME
- https://huggingface.co/datasets/psychemme/PsycheMME-Training

## Pinned release and actual contents

Training repository revision: `3b80a47473802b160b54e760d5dca070efd96f4c`.
Metadata and annotations were downloaded to
`/ocean/projects/cis260338p/apatawar/social_norm_rl/raw/psychemme/3b80a47473802b160b54e760d5dca070efd96f4c/`.
The download manifest records SHA256 hashes. No media was downloaded for
this assessment and no upstream preparation code was executed.

Actual current files contain 3,638 SFT examples and 2,827 GRPO-pool examples.
The paper and some README/config tables still say 3,657 and 2,842. The README
introduction uses the current counts. Do not silently conflate these releases.
There are 1,911 metadata episodes, 1,903 used by the current SFT records, and
427 distinct `(source_dataset, source_video_id)` pairs in the episode metadata.
There are 3,526 single-choice and 112 open questions; most choice questions
have ten options. All SFT examples name 32 training frames.

| Capability | Current questions | After EgoNormia source exclusion |
|---|---:|---:|
| Grounding (G) | 416 | 395 |
| Psychological understanding (W) | 599 | 593 |
| Need (N) | 179 | 174 |
| Timing (T) | 660 | 644 |
| Verbal response (S) | 546 | 531 |
| Embodied action (A) | 648 | 630 |
| Risk/boundary/autonomy (R) | 358 | 344 |
| Memory (M) | 232 | 223 |
| Total | 3,638 | 3,534 |

Sources before exclusion: EgoCom 1,715 questions, Ego4D 855, YouTube Interview
738, YouTube Vlog 130, EasyCom 94, MERCI+ 65, CASTLE 41. ASR metadata contains
103 records. Dialogue and history dependence must be checked rather than
assuming all questions are answerable from sampled frames alone.

The original SFT targets all contain exactly one non-null observation slot:
G 2,796, A 412, W 387, T 32, N 9, R 2; S and M never contain a non-null
target explanation. A question can concern an action or risk while its
target supplies a G observation. This is answer-and-observation supervision,
not eight annotated reasoning steps. The source describes automated question
construction and separate filtering of training data; do not ascribe the
benchmark's unanimous five-expert review to every training example.

## Leakage and shortcuts found

Only the existing ID-only `data/manifests/splits.json` was used for the
EgoNormia overlap check. No reserved question, label, prediction, or video
was inspected.

- The 400-ID development panel shares three Ego4D source videos with six
  PsycheMME episodes containing eight current SFT questions.
- The reserved final partition shares 17 source videos with 50 episodes
  containing 96 current SFT questions.
- Excluding every training episode from these 20 source videos removes 104
  questions and leaves 3,534 candidates. Exact source-ID exclusion does not
  certify absence of alternate uploads or recordings from the same session.
- A is correct in 1,768/3,526 single-choice questions (50.1%); B in 912
  (25.9%). Deterministic random option permutations with remapped answers
  are required. Check references to option letters in question/target text.
- For source validation, group all clips of a source video together, and
  additionally group EgoCom camera views by day/conversation. Investigate
  equivalent recording/session identifiers for EasyCom and other sources.
  A random question split is unsuitable.

The supplied 4B adapters may have trained on the overlapping videos. They
are unsuitable as clean external-training controls for our EgoNormia claim
without their own training-overlap audit. Train our own filtered adapter.

## Media, license, and V100 feasibility

The README estimates a full download around 59 GB. The Hub file metadata
reports `media/frames.tar` alone as 596,674,560 bytes, SHA256
`beaa87b53b33acc6788a18f43250b39f6dcfc20e963115c9637dd75440de2674`.
Using the released frames would avoid downloading all original video clips
for the first SFT study. Verify archive members and image integrity, and
retain only eligible frame paths in the resulting training manifests.

Source-specific terms apply. The repository's Apache-2.0 code license is
not a blanket media license. The data LICENSE assigns CC BY 4.0 to original
numeric results and source-index metadata and links to source-specific
notices (including Ego4D, EgoCom, CC BY-NC, and CC BY-NC-SA sources).
Keep media and source annotations on Ocean; do not redistribute them in
the team code repository.

Our existing Qwen3.5-9B NF4/FP16 language-LoRA setup is the starting hardware
plan. The source configuration uses a 4B model and BF16, so it cannot simply
be copied to a V100. Measure one forward/backward pass on the 32 frames at
their released resolution; if needed, reduce frames or use accumulation
and record the deviation. Do not promise a memory footprint or runtime
before measuring. Frame-list input will also require a deliberate adapter
to the current native-video input code.

## Proposed next experiments

1. Complete a small deterministic visual quality audit, source/session
   partition, answer-permutation validation, and one-V100 training smoke
   test. Pin all changes before EgoNormia scoring.
2. Train **R3-SFT -> PsycheMME-SFT** on the filtered training split, with
   assistant-only loss on answer plus the supplied observation. This keeps
   the existing R3 work in the main pipeline and tests better-aligned
   supervision. Keep public ASR/history as explicitly documented inputs;
   inspect whether the observation is supported by the available modalities.
3. Train **Base -> PsycheMME-SFT** as a control to identify whether R3
   preadaptation helps. Existing Base and R3-SFT checkpoints provide the
   other comparisons. An additional matched R3-SFT continuation can separate
   data-source benefit from extra training exposure.
4. Select training checkpoints using source validation, evaluate the fixed
   400-ID EgoNormia development panel at 1,024 tokens with native metrics,
   and check R3 retention. Report source and modality coverage explicitly.
   Keep the final EgoNormia partition reserved until the final protocol is
   frozen. A later answer-only SFT ablation can test the observation target.
5. If SFT shows useful transfer, test SFT -> RL against continued SFT from
   the same checkpoint. Use a frozen SFT reference for KL, examine reward
   variation before scaling, and avoid treating a correct option as proof
   of a faithful explanation. Expanded PsycheMME use must be described as
   an extension of the original R3-only training study.

## Reproduce the completed metadata audit

`scripts/audit_psychemme.py` verifies local metadata checksums, counts actual
records, reads only EgoNormia split IDs, and emits the audit plus eligible
and excluded SFT IDs. These candidates are not yet train/validation splits.

```bash
python scripts/audit_psychemme.py \
  --root /ocean/projects/cis260338p/apatawar/social_norm_rl/raw/psychemme/3b80a47473802b160b54e760d5dca070efd96f4c \
  --ego-split-ids /ocean/projects/cis260338p/apatawar/social_norm_rl/data/manifests/splits.json \
  --output /ocean/projects/cis260338p/apatawar/social_norm_rl/data/psychemme/audit
```

The completed audit summary is copied to `reports/results/psychemme_audit.json`.
