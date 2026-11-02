# Correction to PsycheMME training-frame representation

During overnight monitoring on 2026-11-02, an input audit found a mistake
in our PsycheMME preprocessing: the generic `frame_paths` native-video path
assigned one-second frame indices on a 32-frame timeline to the released
frame archive. It had no actual frame timestamps. The 1,911 released clip
metadata durations range from 19.5 to 90.04 seconds; only two are within
one second of 32 seconds. Thus these synthetic timestamps are not reliable
physical times. This affects the PsycheMME adapters trained with that path,
not the native EgoNormia evaluation videos or R³ training-video path, which
sample using the actual source frame rate and indices.

A source-video check confirms the problem. For released episode
`-gbqkZfZXUA_e001_2688_3888`, the provided frame at ordinal 31 closely
matches the source clip around 19.23 seconds, whereas our path assigned
31 seconds. Other released frames likewise match different actual times.
The source image/video visual identity was checked. Thumbnail matching is
an approximate alignment check, not certified exact timestamp annotation;
we therefore do not reconstruct new clock times by interpolation. The
machine-readable audit is `results/psychemme_frame_timing_audit.json`.

Existing scores and weights are retained as legacy-representation results.
They measure those models accurately, but a negative transfer result from
that representation is insufficient to conclude that the dataset itself
is unhelpful. The first-option bias remains an observed failure, with its
causal mechanism unresolved. Original SFT/RL comparisons on R³ remain
separate from this representation error.

The correction follows the released training interface: encode the original
32 chronological images as images, with a maximum side of 256 pixels,
and provide no invented frame timestamps. Public transcript timestamps
remain as supplied. This also restores the 32-image coverage in the source
release. Targets, eligible example IDs, source exclusions, learning rate,
seed, and one-pass training stay fixed. The experiment changes both the
image/video representation and frame count relative to our eight-frame
legacy path; it does not isolate their separate effects.

New 9B and 27B normal/long-prompt smoke tests verify memory and optimizer
behavior before corrected training is allowed. Preparation job 50121951,
9B smoke 50121952, 27B smoke 50121953. Standalone corrected SFT jobs
50121970/50121971 depend on their smoke test. These new runs use distinct
names ending in `_images_v2`. The not-yet-started R³→PsycheMME stages are
redirected to this corrected representation, with a preflight gate; their
original specs are archived as `.legacy.json`. The R³ parent training
stages and completed legacy checkpoints are unchanged.

The original full-study registry and boundary hash are retained. An
explicit dated `registry_amendments.json` records the corrected experiment
specs and superseded stages; its copy is
`results/frame_images_recipe_amendment.json`. These are technical corrections
made after some full benchmark results were visible, so their results must
be reported as amended/post-freeze experiments rather than a new untouched
benchmark test. No EgoNormia example, label, or rationale enters training.
The same mandatory exclusion guard applies to all 3,534 training examples.
