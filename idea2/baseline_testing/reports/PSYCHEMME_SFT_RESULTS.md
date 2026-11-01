# PsycheMME SFT transfer: completed exploratory result

The two predeclared PsycheMME SFT runs and their dependent source validation,
EgoNormia evaluation, and paired analysis all completed successfully on
2026-11-01 (Slurm jobs listed in `PSYCHEMME_SFT_PROTOCOL.md`). No EgoNormia
example or answer entered training. Both conditions used the pinned
PsycheMME training release, the same 3,032 source-grouped training questions,
eight sampled video frames, option permutation, answer-plus-observation
targets, 300 accepted updates, and one V100-32 GB each. The 389-question
source validation split is disjoint by recorded source/session group. The
1,025-example EgoNormia final partition remains untouched.

| Model | Ego action | Ego justification | Ego joint | Ego sensibility IoU | Invalid | PsycheMME val120 |
|---|---:|---:|---:|---:|---:|---:|
| Original Qwen3.5-9B | 48.0% | 41.8% | 31.2% | 48.8% | 22 | — |
| R³-FDT answer-only SFT | 50.0% | 41.8% | 35.2% | 48.7% | 17 | — |
| PsycheMME SFT from base | 48.2% | 43.8% | 36.8% | 48.8% | 7 | 75.0% |
| R³-FDT SFT → PsycheMME SFT | 50.7% | 45.2% | **38.8%** | 49.2% | 7 | 77.5% |

All EgoNormia values use the exact same 400 supplied IDs from 348 source
videos, correct preceding videos, and a 1,024-token generation cap. Failed
and invalid examples remain in all denominators. Source validation is the
same first 120 IDs sorted by ID, covering 16 source groups; it is a
diagnostic subset rather than the full 389-example validation set.

The combined model's joint gain over the original checkpoint is +7.5
percentage points (95% source-video-cluster bootstrap interval +3.3 to
+11.8); over R³-SFT it is +3.5 points (0.0 to +7.0); over PsycheMME SFT
from base it is +2.0 points (−0.8 to +4.8). Relative to R³-SFT, it turns
36 joint failures into successes and 22 joint successes into failures.
These intervals reflect sampling over the 348 source videos, **not**
training-seed variability. The lower interval endpoint for the main
R³-SFT comparison is effectively zero, so a repeat is needed before claiming
a reliable incremental benefit.

The two training jobs completed 300 accepted updates each. The base start
visited 307 training questions (seven logged nonfinite-gradient skips)
and took 14:17 in Slurm; the R³-SFT start likewise visited 307 and took
19:17. Peak GPU memory reserved was 12.19/12.67 GB. The source validation
accuracy was 90/120 versus 93/120. Training adapters remain on Ocean at
`runs/psychemme_sft_base/checkpoint-00300` and
`runs/psychemme_sft_r3sft/checkpoint-00300`.

The new SFT run improves joint action-and-justification accuracy but does
not materially change sensibility IoU. The combined model still gets neither
action nor justification correct on 171/400 examples; another 48 have only
the action right and 26 only the justification right. It is weakest in
Politeness (27.2% joint) and Communication/Legibility (30.6%), while Safety
reaches 45.1%; categories overlap, and these are descriptive subsets. The
correctly parsed output count improved (invalid examples 17 → 7 compared
with R³-SFT), which may explain some of the joint gain. Only four
justification outputs hit the 1,024-token cap for the combined model.
Neither a matched extra R³-SFT continuation nor a no-video control has yet
been run for the new adapters, so the present study does not isolate extra
training time from PsycheMME's data content or prove stronger visual
grounding. The repeated 400-ID panel is a development panel and the
results have one training seed. Final model selection should be frozen
before the reserved source-video-disjoint EgoNormia evaluation.

Machine-readable results are in `reports/results/psychemme_*`.
Raw predictions and training logs remain on Ocean.
