# Epoch SFT, replication, controls, and 27B feasibility

Protocol fixed before results on 2026-11-01. See
`results/next_experiment_jobs.json` for submitted job IDs and output paths.

The 300-update PsycheMME study visited only about one tenth of its 3,032
training questions. This study trains new adapters from the original model
or the same frozen R³-SFT-100 adapter for 3,032 accepted updates. Seeds 777
and 1777 independently shuffle the source examples and option order; the
R³-SFT initialization itself remains the existing seed-777 checkpoint.
This replicates the continuation stage, not the complete upstream pipeline.
All other SFT hyperparameters and eligibility exclusions match the pilot.
Finite-gradient skips mean visits may slightly exceed one epoch.

Save checkpoints at steps 750, 1500, 2250, and 3032. For each run, select
highest answer accuracy on the same frozen sorted 120-question PsycheMME
validation panel as the pilot, breaking ties toward the earlier step.
This is a source validation subset, not a new test set. Save every candidate
score and the selection to `selected.json`. Then evaluate the selected
adapter on the existing EgoNormia 400 development IDs at 1,024 tokens and
on accessible R³ test questions at 192 tokens for retention. Final EgoNormia
source-video groups remain reserved. Analyze source-video-cluster paired
intervals and show both seeds individually; two seeds give only a limited
estimate of training variability.

As a data-source control, continue the same R³-SFT checkpoint for 3,032
additional supervised updates on accessible R³ training data, with learning
rate 5e-6 and the same rank-8 adapter architecture. This matches update and
example counts, not token counts or GPU hours: the R³ targets are shorter,
its question pool is smaller and reused, and it has no observation targets.
Evaluate its fixed final checkpoint on the same EgoNormia panel. The control
starts in job 50106860; the initial queued job 50106839 was cancelled before
running to match the learning rate.

For visual controls, use the first 100 supplied EgoNormia IDs in their
existing order. Evaluate blind and mismatched-video conditions for each
selected epoch adapter and the original model. Blind keeps the ordinary
video-task wording with video omitted (`--blind-same-prompt`); this avoids
the extra blind-context instruction used in earlier controls. Mismatches
are deterministic over the same panel. The existing correct-video
predictions provide the paired reference. Justification remains conditioned
on the model's selected action; the control therefore measures the whole
pipeline, not an isolated justification effect. Controls are developmental,
and video sensitivity alone does not establish faithful reasoning.

Qwen3.8-27B is pinned at model revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Download official original weights
(about 55.6 GB) to Ocean, then quantize locally to NF4 with FP16 computation.
Use a separate `envs/qwen38` environment with Transformers 5.22.1 and the
established V100-compatible Torch/CUDA dependencies. Installation uses
`--ignore-installed` so dependencies imported from the 9B environment are
not uninstalled. The existing 9B environment remains the dependency source.

The V100-32 GB smoke test measures model loading, eight-frame inference
with 128 generated tokens, and one short language-LoRA backward pass on a
PsycheMME training example. It writes the stage, peak memory, response,
loss, finite-gradient status, or exception to
`runs/qwen38_v100_smoke/status.json`. The current model preparation promotes
some frozen parameters to FP32; an OOM at that stage would require a more
memory-efficient preparation before concluding inference is infeasible.
This is a feasibility test, not a benchmark. A 27B EgoNormia baseline and
27B SFT are contingent on successful hardware and implementation checks;
they are not yet queued. Its 9B counterpart's adapters cannot be reused.
The long-reasoning RL comparison remains a subsequent stage after assessing
these supervised learning curves and the reward variation of the selected
policy.
