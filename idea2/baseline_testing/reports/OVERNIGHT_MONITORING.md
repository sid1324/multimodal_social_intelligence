# Overnight monitoring — 2026-11-02

The user requested continued autonomous work and checks every 15 minutes.
The agent performs monitoring passes with `scripts/monitor_full_study.py`,
which records timestamped training/evaluation progress on Ocean. Completed
models whose previously missing media have become available are repaired
through a new resumable inference pass, preserving the original predictions.
No training input or frozen recipe is changed based on full benchmark labels.

Initial check:

- Full eligible PsycheMME training is progressing for both 9B and 27B.
- The 9B expanded R³ SFT has started; 27B expanded R³ SFT is queued.
- Acquisition attempted all 2,812 R³ videos, but only 373 were accessible,
  covering 6,037 questions. This is expanded accessible-data training,
  not training on all 41,529 annotations. Original unavailable videos,
  YouTube sign-in/bot checks and HTTP errors block the rest.
- EgoNormia has 1,586/1,853 validated clips. Remaining acquisitions mostly
  failed on HTTP 429. Recovery job 50118192 uses one serial stream,
  respects Retry-After, and applies a 15-minute cooldown before retrying.
- Full evaluations keep missing-video records in their denominators.
  Following recovery, completed evaluations are repaired and reanalyzed.
- Source-video exclusions remain mandatory for every training session.

The official R³ repository still links to the annotation release, and the
CondensedMovies challenge's additional download is a feature archive rather
than raw training video. It does not resolve the missing-media blocker:
https://github.com/LiXingNiu/Read-the-Room and
https://github.com/m-bain/CondensedMovies-chall#dataset-download.

Progress snapshots and recovery logs remain in
`/ocean/projects/cis260338p/apatawar/social_norm_rl/data/full_study/`.

At 06:08 UTC the serial recovery completed: all 267 rate-limited clips
were downloaded and validated, giving 1,853/1,853 EgoNormia preceding clips.
There were no failed attempts in the recovery stream. Missing-media
predictions from the earlier incomplete coverage will be rerun before the
completed full-model summaries are accepted.

At 06:43 UTC, the full 27B baseline array throttle was raised from two to
six simultaneous one-V100 shards. The two not-yet-started 400-only baseline
shards (50115599_2/3) were cancelled because the user expanded the objective
to all 1,853 examples; already-running 400-only shards and all saved
predictions were retained. No unrelated jobs were changed.

At 06:59 UTC, the 9B full-PsycheMME model completed all 3,534 eligible
examples. The first completely repaired historical model evaluation
(epoch base start, seed 777) has 1,853/1,853 media-present predictions.
Its original 400-ID score exactly reproduces 35.75%, supporting pipeline
consistency. Full joint accuracy is 33.78%; the new-source cohort is 32.68%.
No improvement claim is made until unmodified baselines finish. Complete
results are collected in `FULL_BENCHMARK_RESULTS.md` and its JSON companion.

At 07:31 UTC both 9B and 27B full eligible PsycheMME training stages were
complete. The 9B model has a clean full-benchmark score of 26.61% joint,
with 31.9% action accuracy and only four invalid examples. It selects action
index zero in 1,514/1,853 cases (81.7%), although the actual permuted source
training targets are not dominated by their first position. This is an
observed answer-position bias; its training cause is unresolved. The fixed
R³ and 27B comparisons remain in progress. No new training choices have
been made from these full benchmark labels.
