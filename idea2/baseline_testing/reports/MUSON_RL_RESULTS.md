# Post-midterm MUSON action RL pilot: measured outcome

The [protocol](MUSON_RL_PROTOCOL.md) fixed a 50-update, class-balanced RLOO
continuation from the R³-FDT RLOO-100 adapter. MUSON is an independent
egocentric **image** dataset with six short-horizon navigation actions; it
was a secondary source, and no EgoNormia data or labels entered training.
The public v1.0 archive and split are recorded in
[the data audit](MUSON_AUDIT_AND_PLAN.md). The 1,300-image MUSON test split
and the source-video-disjoint EgoNormia final partition remain untouched.

The continuation completed on one V100-32 GB in 2:21:34 (`50094260`). It
visited 178 images, accepted 50 mixed-reward updates, and skipped 128 groups
whose four sampled completions had identical rewards. Mean sampled
correctness was 18.8% across the six balanced action classes, near the
16.7% uniform-choice rate. No nonfinite-gradient updates were logged;
peak GPU reserved memory was 13.08 GB. The adapter is
`/ocean/projects/cis260338p/apatawar/social_norm_rl/runs/rloo_muson/checkpoint-00050`.

| Metric | R³-FDT RLOO starting adapter | MUSON RLOO continuation |
|---|---:|---:|
| MUSON validation accuracy, fixed 300 images | 51.0% | 52.7% |
| MUSON predictions of “move forward” | 247/300 | 251/300 |
| EgoNormia action accuracy, 400 items | 48.2% | 48.8% |
| EgoNormia justification accuracy, 400 items | 44.0% | 43.0% |
| EgoNormia joint accuracy, 400 items | 33.8% | 32.0% |
| EgoNormia sensibility IoU, 400 items | 49.0% | 49.2% |
| EgoNormia invalid examples | 21 | 16 |

The 300-image MUSON validation panel contains 158 “move forward” labels, so
an always-forward predictor would score 52.7%. The starting model achieved
88.6% on forward images but only 6.6% on forward-right, 13.7% on
forward-left, 11.8% on stop, and zero on both turn classes. After RL,
forward-right rose to 13.1%, but stop fell to 5.9%, forward-left stayed
13.7%, and both turns stayed at zero. The model continued to choose
forward on 83.7% of images. Thus, the small overall gain does not indicate
that it learned the safety-critical rare actions.

All 400 requested EgoNormia items were evaluated with correct preceding
videos and a 1,024-token output cap (`50094261`); the analysis job
`50094262` completed. The paired MUSON-continuation-minus-R³-FDT-RLOO
joint difference was −1.75 percentage points (95% source-video-cluster
bootstrap interval −3.9 to +0.5). Action changed by +0.5 (−1.5 to +2.5)
and justification by −1.0 (−3.6 to +1.8). The continuation also scored
below the saved SFT control (35.2%) and only +0.75 points above the
unmodified model (31.2%). These are single-seed exploratory estimates on
the previously used 400-ID development panel.

This experiment does not support an EgoNormia transfer benefit from 50
exact-action RLOO updates on MUSON. It also demonstrates a sparse-reward
failure: class-balanced sampling exposed non-forward scenes, but many groups
had no reward variation and the adapted model remained forward-biased.
The dataset's first-person viewpoint and action target are closer to
EgoNormia than the prior NormLens pilot, yet it uses one static image,
navigation-only decisions, and a single preferred action. A stronger next
test should first teach the action and explanation format with supervised
learning, then ask whether RL adds value beyond that warm start, using a
reward that tests action and explanation consistency. The current result
does not justify extending the same sparse-reward run blindly.

Machine-readable summaries are in `reports/results/muson_*`; raw
predictions, training logs, data, and the adapter remain on Ocean.
