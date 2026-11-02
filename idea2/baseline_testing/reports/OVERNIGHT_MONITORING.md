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
