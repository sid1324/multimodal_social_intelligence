# Video social reasoning RL — CMU 11-777

R3-FDT trains a 4-bit Qwen3.5-9B LoRA policy. EgoNormia measures transfer to
normative decisions. The midterm's 400-ID correct-video panel measures joint
accuracy of 23.8% base, 26.2% RLOO, and 28.0% answer-only SFT. RLOO's paired
gain over base is uncertain (95% source-video-cluster interval −0.8 to +5.7
percentage points), so an RL-specific transfer benefit has not been shown.
See the [paste-ready midterm section](reports/MIDTERM_SECTION.md),
[full research record](reports/MIDTERM_DRAFT.md),
[aggregate results](reports/results/video400_summary.json),
[frozen protocol](reports/PROTOCOL.md), and [data audit](reports/data_audit.json).
A separate 384-token diagnostic on the same 400 items gives 30.8% base,
33.5% RLOO, and 35.0% SFT joint accuracy; see its
[paired comparisons](reports/results/cap384_video400_paired.json).

## Environment and storage

Code: this directory (`idea2/baseline_testing` in the team repository).
Large artifacts: `/ocean/projects/cis260338p/apatawar/social_norm_rl`.
The latter can be overridden with `SOCIAL_RL_STORE`.
No credentials are embedded. Existing unrelated environments/jobs are untouched.

Python 3.12, PyTorch 2.8.0 and torchvision 0.23.0 from the CUDA 12.6 index,
then `pip install -r requirements.txt` and `pip install -e .`.
`bash scripts/bootstrap_data_environment.sh` creates the annotation/media
environment, and `bash scripts/bootstrap_environment.sh` creates the model
environment with these pins. Both write READY markers only after import checks.
Use a V100-compatible CUDA build, not a CUDA 13 build that drops SM70.
The exact installed dependency freeze is saved to reports/environment-lock.txt.
Slurm scripts request one V100-32 on GPU-shared, account cis260338p.
Submit them from this directory; Slurm uses that submission directory to find
`slurm/env.sh` and write logs under `logs/`.

## Data

Run `python scripts/bootstrap_sources.py` to verify or fetch the pinned R3 and
EgoNormia annotations, CondensedMovies movie metadata, and the exact benchmark
repository commit. Raw files are in `data/raw` under storage.
`scripts/prepare_data.py` normalizes them and freezes movie-disjoint R3 splits,
plus source-video-disjoint midterm/final EgoNormia partitions around the exact
IDs in `egonormia_400_sample_ids.txt`. The final
partition is explicitly rejected by the midterm evaluation script.
`scripts/acquire_media.py r3 --limit 100` and
`scripts/acquire_media.py ego --limit 400` fetch the team's fixed 400-ID panel.
Failed acquisitions are logged. R3 downloads are video-only at up to 480p;
sampling is limited to the released clip duration to exclude advertisement
outros. EgoNormia uses `video_prev.mp4`, never the future `video_during.mp4`.

R3 annotations are CC BY-NC 4.0. Movie rights remain with their owners.
EgoNormia release attribution/terms apply; its MIT code license is retained in
licenses/. Do not redistribute source video bytes in this repository.

The pinned model snapshot is Apache-2.0. On the installed Transformers 5.3.0,
the eight selected frames are presented as four ordered two-frame video
segments with their source timestamps: one eight-frame segment gives four
video-token runs but only one position-grid row and fails generation. The
successful one-V100 smoke run and this compatibility decision are recorded in
`reports/STATUS.md`.

## Commands

```bash
source slurm/env.sh
python -m pytest -q
sbatch slurm/smoke.sbatch
sbatch slurm/evaluate.sbatch \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/base_ego" --conditions video blind mismatch
sbatch slurm/train.sbatch \
  --manifest "$SOCIAL_RL_STORE/data/manifests/r3_train_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo" --steps 100
```

Do not run GPU work on a login node. A model snapshot `READY` marker is required;
`scripts/finalize_model.py` verifies the pinned download. The smoke test uses
synthetic inputs and one real R3 training clip strictly for engineering, never
as a reported experiment.
For a fresh setup, run `$SOCIAL_RL_STORE/envs/data/bin/python scripts/download_model.py`
from the repository after setting
`SOCIAL_RL_STORE` (the script forces HF cache to Ocean), then
`$SOCIAL_RL_STORE/envs/data/bin/python scripts/finalize_model.py`.
This project's completed snapshot was fetched on a CPU compute node using
verified HTTP byte ranges after Hugging Face Xet streams stalled on the login
node; see `HANDOFF.md` and `slurm/download_model_cpu.sbatch` for the actual
provenance and checksums.

Evaluation appends resumable predictions. Reinvoke the same command to continue
after walltime. Do not change settings or the manifest in an existing run.
Training checkpoints include optimizer, loss scaler, data cursor, and random
states; resume using the same command plus `--resume`.

```bash
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego" \
  --output reports/generated/baseline
python scripts/failure_counts.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --run "$SOCIAL_RL_STORE/runs/base_ego" \
  --output reports/generated/baseline
```

For official scorer comparison, run its unmodified `compute_scores_final.py`
from a generated results directory with `--file official`.

## Implementation choices

The pilot uses RLOO (leave-one-out on-policy policy gradients), four sequential
rollouts, strict correctness reward, fixed-cap token loss normalization, and a
sampled reference KL penalty. This avoids a critic and a separate generation
server. Zero-variation groups are logged and skipped; their fraction is a key
training diagnostic. This is a deliberately small initial RL implementation,
not a reproduction of a particular GRPO paper. `--method sft` provides an
answer-only supervised control using the same dataset/model/adapter targets.

Official EgoNormia prompt templates are extracted verbatim from the pinned
repository. Justification is conditioned on the **predicted** action. Native
last-integer parsing is preserved, with stricter final-line validity logged
alongside it. Unlike upstream's dropped failed examples, attempted examples
remain in our denominator. Raw outputs, invalid rates, input/output token
counts, memory, and elapsed time are retained. Results are not comparable to
full-benchmark leaderboard scores unless coverage and input protocols match.

## Research status

See reports/STATUS.md for actual jobs, checks, results, and remaining blockers.
Never treat successful setup, a synthetic smoke test, or training reward as
evidence of EgoNormia improvement.
