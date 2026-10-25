# Midterm execution handoff — updated 2026-10-24

## Current state

All planned midterm runs finished. The three correct-video models each have
400/400 predictions on the exact supplied EgoNormia IDs (348 source-video
groups); base/RLOO/SFT joint accuracy is 23.8/26.2/28.0%. RLOO minus base is
+2.5 percentage points (source-video-cluster bootstrap 95% interval −0.8 to
+5.7); SFT minus base is +4.3 (+0.5 to +8.0). The 140-ID no-video and
different-video probes remain separate. See `reports/MIDTERM_SECTION.md` for
the paste-ready section, `reports/MIDTERM_DRAFT.md` for detailed analysis,
and `reports/results/video400_provenance.json` for merge hashes.

A separate 384-token diagnostic on the same 400 correct-video items also
finished: base/RLOO/SFT joint scores 30.8/33.5/35.0%. The longer cap adds
about seven points for every model and sharply reduces invalid outputs;
RLOO versus base remains uncertain. Results and provenance are under
`reports/results/cap384_video400_*`. This was motivated by observed
truncation and does not replace the original 192-token primary protocol.

Jobs 49766293, 49766294, 49766295, and 49766301 all completed successfully.
Cap diagnostic jobs 49771983, 49771986, 49771988, and 49771990 also completed.
The final RLOO adapter is on Ocean at `runs/rloo/checkpoint-00100`; SFT is at
`runs/sft/checkpoint-00100`. Generated summaries are under
`reports/generated/video400/` and raw predictions under Ocean `runs/`.

## User decisions and priorities

Deadline is about two days away. This workstream is RL only; other teammates
cover other ideas. Use Qwen3.5-9B with NF4 4-bit loading and language LoRA,
initially one V100-32 on PSC GPU-shared. User authorized implementation,
downloads, experiments and Slurm submission. Prioritize real baseline results
and multimodal error analysis over unmeasured claims of improvement.

**User changed storage/account to cis260338p.** All project artifacts now live
at `/ocean/projects/cis260338p/apatawar/social_norm_rl`; code stays at
`/jet/home/apatawar/multimodal_project`. The former project storage directory
was moved, not left as a duplicate. Paths inside manifests and both venvs were
relocated; group ownership is cis260338p. Do not use the former destination.

## Actual progress

- Code, protocol, report draft, Slurm scripts and eight CPU protocol tests exist.
  All eight tests passed after freezing the 400-ID panel. Shell syntax checks passed.
- Official pinned annotations downloaded and audited: R3 has 41,529 questions
  over 2,812 videos. Ego HF release has 1,743 rows versus repository 1,853.
- The original 200-clip pool was superseded by the user-supplied fixed 400-ID
  panel. All 400 current preceding clips were downloaded and decoded. The
  accessible manifest matches the supplied ID file exactly in order.
- Initial R3 acquisition finished: 96/100 selected videos validated, yielding
  1,262 training, 86 validation and 79 test questions with media. Four failed
  acquisitions are recorded.
- The pinned model weights and isolated training environment are ready. CPU
  job `49742706` verified all four shard SHA-256 digests and wrote `READY`.
- The first V100 smoke job `49730226` loaded the model but failed before
  inference due to a Qwen3.5 processor video-grid mismatch. CPU preflight
  confirmed the fix; replacement smoke `49743943` passed on one V100-32 GB.
  It confirmed real R3 video generation, finite adapter gradients, optimizer
  step and adapter reload parity at 13.01 GB peak reserved memory.
- Initial base-model EgoNormia scoring completed on 140/400 supplied IDs (133 source
  videos): joint accuracy 28.6% with correct video, 10.0% no video, 7.1%
  mismatched video. R3 held-out baseline: 62.5% on 40 validation questions,
  68.3% video and 61.7% blind on 60 test questions.
- RLOO completed 100 optimizer steps from 156 sampled R3 training groups
  in 84.1 minutes; final adapter is `runs/rloo/checkpoint-00100` on Ocean.
  Adapted EgoNormia transfer is reported in the current-state section above.
- Workspace is a local Git repository. Commit `8e763d1` prepared the
  400-ID extension; later report updates should be committed after review.

## Running setup after relocation

The Slurm chain in `logs/midterm-jobs.txt` has completed. Project GPU jobs
were run sequentially on one V100. The CPU low QoS was test-submitted on
this account; normal RM QoS was unavailable.
CPU baseline-analysis job `49743947` completed after baseline; it wrote
objective Ego failure counts and R3 held-out summaries.
Response-cap diagnostics 49754768 and 49764791 completed on the first 40
fixed panel IDs. They are separate from the primary 192-token comparison.
Answer-only SFT control 49754875 completed 100 nominal updates and its
400-ID correct-video evaluation finished in job 49766295.

Check processes and logs before starting duplicates. Initial detached nohup
attempts exited with their parent tool session; the successful CPU download
was run through Slurm:

| Work | Exec session | Log |
|---|---|---|
| `bash scripts/bootstrap_environment.sh` | 5308 | `logs/environment-setup.log` |
| `scripts/download_model_ranges.py`, then `scripts/finalize_model.py` | CPU Slurm job 49742706 | `logs/model-download-cpu-49742706.log` |
| `scripts/acquire_media.py r3 --limit 100` | completed | `reports/r3_media.log` |
| `scripts/acquire_media.py ego --limit 400` | completed | `reports/ego_media.log` |

`logs/setup-pids.txt` describes the exited initial attempt and is obsolete.
Training environment and model have separate READY markers under storage.
Neither marker should be fabricated. Model download reused partial shard files.
Eight completed CUDA dependency wheels were salvaged into cache/wheelhouse.
Model shard fetching was restarted with two workers after all four parallel
streams stalled for several minutes. The Xet client appears to truncate an
incomplete shard when it starts that shard again. After a second multi-minute
stall, Xet was stopped. The login-node HTTP fallback also stalled; a CPU
compute-node 4 MiB probe completed at about 10.8 MB/s. Job `49742706` ran
the SHA256-verified HTTP-range fallback and `finalize_model.py` successfully.
The untouched partials
were also copied to `cache/model_partials/shard-00003.prefix` and
`shard-00004.prefix`. When the active streams paused again, shards 1 and 2 were
saved as `.prefix` files there too. All four saved prefixes passed first/last
byte comparisons against official HTTP ranges. The two active shards had
restarted from zero. The download completed at about 23:29 EDT; this is not a
model result.
Ego media uses bounded HTTP byte-range retries because long streams stalled;
a repository-only sample video was downloaded and decoded successfully.
The action prompt was re-extracted from the official evaluator after a preflight
check caught an accidental retrieval-only suffix; all three prompt tasks are
covered by a regression test. No GPU prediction was made with the bad prompt.
The first GPU smoke loaded Qwen3.5-9B in ~231 seconds, then generation raised
`StopIteration` in `get_rope_index`: one eight-frame video produced four
video-token runs but only one `video_grid_thw` row. The current input path
presents the same eight sampled frames as four ordered two-frame video
segments with timestamps, so the processor yields four matching rows. CPU
preflight checked all three Ego prompt types. Replacement smoke passed.

Use `envs/data/bin/python` for acquisition; `envs/qwen35/bin/python` for training.
Both are under the new storage root. Set `PYTHONPATH` to workspace `src` and
`LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib` (prepend if already set).
Acquisition needs online HF access; `slurm/env.sh` intentionally sets offline
mode for GPU jobs. Pin revisions from configs/midterm.json. Full official weights
are downloaded and quantized at load, not a third-party quantized checkpoint.

## Next execution steps

1. The setup and READY markers have passed. Pinned torch 2.8.0 CUDA12.6
   contains sm_70. An existing unrelated torch
   2.11 CUDA12.9 environment lacked sm_70 and must not be substituted.
2. Verify available-media manifests after acquisition finishes. R3 eligible
   manifests are rebuilt at completion, so may be empty while download runs.
3. Replacement smoke job `49743943` passed on a V100-32 GB. Inspect
   `runs/smoke/result.json` in Ocean for the engineering measurements.
4. Debug version/API, precision, memory and hybrid DeltaNet fallback throughput
   as needed. Code uses SDPA and the PyTorch fallback, not assumed V100 support
   for optional flash kernels. Check vision processor and LoRA target matches.
5. Run `slurm/baselines.sbatch`: video / blind / mismatched-video Ego conditions,
   plus held-out R3 evaluation. Monitor throughput; report actual coverage.
6. Run a short real-data RL pilot before the planned 100-step RLOO job. See README
   for train arguments. Inspect rewards, varying-reward group fraction, KL,
   gradient finiteness, memory and steps actually completed. SFT is optional.
7. Run `slurm/adapted.sbatch` only after baseline and usable adapter exist.
   Compare identical completed IDs, report native metrics, paired source-cluster
   uncertainty, categories and Verified subset coverage.
8. Inspect generated error-review HTML/contact sheets and raw responses. Write
   observed successes/failures into reports/MIDTERM_DRAFT.md with real tables and
   multimodal hypotheses; never infer that output text reveals internal reasoning.

Slurm scripts use account cis260338p and one v100-32. Do not change/cancel unrelated
user `night-*` jobs (49345706, 49345708, 49345710, 49345709 at handoff).

## Scientific and implementation constraints

R3-FDT trains; EgoNormia never trains. R3 splits are movie-group disjoint using
CondensedMovies imdbid. Ego source-video groups are split around the team's
fixed 400-ID panel: 828 midterm-group rows and 1,025 reserved final rows in
the repository release. Recheck the accessible manifest after acquisition.
Two publicly previewed groups were explicitly assigned to midterm. Evaluation
rejects reserved final examples. See reports/PROTOCOL.md and data_audit.json.

Use Ego `video_prev.mp4` only. Justification is conditioned on the model's
predicted action. Official prompts/order/native parsers are preserved; stricter
validity and errors are separately reported. Attempted failures stay in our
denominator. Joint accuracy is per-example action AND justification correctness.
R3 movie footage tests descriptive causal reasoning; normative transfer remains
an empirical hypothesis. Record access gaps and respect source-media rights.

RLOO uses four sequential sampled responses, strict multiple-choice correctness
reward, leave-one-out advantage, frozen-base sampled KL and language LoRA.
Zero-reward-variation groups are skipped and logged. No critic or separate rollout
server. Checkpoint includes optimizer/scaler/RNG/data cursor. Eight frames at
max side 384, max 192 generated tokens, rank 8, LR 1e-5, beta .02 are pilot settings.
Evaluation should compare the same quantized frozen starting model to its adapter.

Read src/social_rl/model.py, scripts/train.py and reports/PROTOCOL.md before
changing the objective. CPU tests cover protocol properties, not GPU correctness.
No changes to model settings within an existing output directory: evaluation
metadata guards against mixing runs. Full benchmark leaderboard comparability
must not be implied for this partial panel/input protocol.
