# Idea 1 midterm: visual regularization and NormLens SFT

This additive package preserves the earlier Idea 1 analysis. It contains selected code for the accepted seed-17 NormLens baseline and targeted/random visual-regularization comparison, not the full ongoing experiment workspace.

## Accepted evidence

Qwen3.5-9B; NormLens train/dev/test 1395/296/291. Accuracy uses 274 untied labels. Vote NLL: frozen .8295911671, dev-calibrated frozen .7792007294, clean SFT .6344327162, H-T .6104434288, H-R .6328820913. Hybrid eta=.5 is a prospective amendment; original eta=.25 potency failure remains part of the study history. Earlier downcast hybrid evaluations are excluded. Results are single-seed and exploratory, not proof of causal grounding or curriculum order. Family tests, further seeds and hybrid input controls are not claimed complete.

## Contents and provenance

`src/` contains the exact CPU scoring/provenance dependency closure used in the saved-output reanalysis. `snapshots/training`, `snapshots/export`, and `snapshots/evaluation` contain selected Python dependency closures extracted byte-for-byte from the immutable source archives bound to the accepted hybrid training/export/corrected evaluation. These are distinct versions; do not combine them and claim byte-identical reproduction. Shared imported utilities may contain branches for other tasks, but no other experiments or their outputs are included. Baseline training utilities in the training snapshot are preserved as dependencies; this package does not assert that this later snapshot is the original baseline launch snapshot.

`SOURCE_MANIFEST.json` records original member names, archive hashes, and delivered file hashes. `configs/` contains scientific-setting excerpts with original config hashes, not executable launches: private data paths, scheduler authority and artifact bindings are intentionally omitted. Runtime requirements are inputs, not a complete frozen GPU environment. Dataset/model acquisition and original runtime pins must be supplied separately for model reruns; this package does not launch training or submit jobs.

## Reproduce the accepted CPU analysis

Install `requirements-cpu.txt` in an isolated environment, then run:

```sh
python reanalyse.py --artifact-root /path/to/original/grounded_norm_sft --out /path/to/new/analysis_output
```

The artifact root must contain the original private labels and hash-bound prediction/export/fidelity artifacts, with recorded paths resolvable. Those data, model weights, credentials and private receipts are not redistributed. The script fails on missing or mismatched inputs and refuses an existing output directory. It reuses the frozen calibration; it does not refit calibration, select a checkpoint, generate predictions or run a GPU job. Paired bootstrap: 2000 resamples, PCG64 seed20261021, whole image clusters (253), example-weighted means, linear percentile 95% intervals. It does not estimate training-seed uncertainty or adjust multiplicity.

`results/accepted_seed17.json` provides the previously recomputed aggregates. No raw images, per-example data, checkpoints, scheduler ledgers, credentials, unrelated runs or unfinished experiment configs are published.
