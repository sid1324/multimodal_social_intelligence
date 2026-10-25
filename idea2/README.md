# Idea 2 — video social reasoning

This directory holds two distinct stages of the reasoning workstream:

- [Data exploration](data_exploration/) contains the earlier Social Genome
  evidence and annotation feasibility study, including its scripts, tests,
  tables, and figures. Its original relative layout is preserved inside that
  directory.
- [Baseline testing](baseline_testing/) contains the R³-FDT video training
  and EgoNormia transfer experiments: Qwen3.5-9B NF4, RLOO, an answer-only
  SFT control, evaluation code, audited aggregate results, and the midterm
  report section.

R³-FDT is used for training; EgoNormia is used only for evaluation. The
baseline-testing README records the exact 400-ID panel, measured results,
model/data revisions, PSC Slurm setup, and Ocean artifact location. Raw media,
model weights, caches, and individual prediction files are not in Git.
