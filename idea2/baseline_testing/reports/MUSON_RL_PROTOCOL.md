# Exploratory R³-FDT → MUSON action RL continuation

The [MUSON audit](MUSON_AUDIT_AND_PLAN.md) records the source, split, rights
uncertainty, and action imbalance. This experiment is fixed after the
NormLens pilot failed to improve EgoNormia and before any MUSON transfer
result is inspected. It tests whether an **egocentric action-choice** reward
transfers better than image-based right/wrong/impossible judgment rewards.
R³-FDT remains the primary video reasoning source and supplies the frozen
RLOO-100 starting adapter. MUSON is a secondary still-image source. No
EgoNormia data or labels enter training.

The MUSON training manifest has 7,710 images in 78 deterministic
100-image blocks. Its six action classes are round-robin sampled in equal
frequency, with a seeded shuffle within each class, to make the 54.5%
forward-action majority less dominant. Options are randomly permuted per
question. The model sees only the current first-person image and the six
generic action choices; the released reference perception/reasoning and
explanation strings are excluded from the prompt and reward. The reward
is one exactly when the generated final action letter matches the released
action class. Four sampled completions form a group, and uniform-reward
groups are skipped. The RLOO advantage is each completion's reward minus
the mean reward of the other three. One update per mixed group uses the same
sampled base-model KL coefficient (0.02), NF4/FP16 frozen Qwen3.5-9B,
rank-8 language LoRA, optimizer, and 192-token training response cap as
the R³-FDT RLOO pilot. The FP16 loss scaler starts at 1024 after the prior
engineering pilot found high initial overflow at its default scale.

Fix **50 accepted updates**, one seed, and the starting R³-FDT RLOO
checkpoint. A two-update MUSON engineering pilot checks image processing,
gradient finiteness, and memory before the main run. The starting adapter
and continuation are evaluated on the same fixed 300-image MUSON validation
subset, selected by a stable hash before training. Overall and per-class
accuracy, invalid rate, cap hits, and action prediction counts are reported.
The 1,300-image MUSON test partition remains untouched at this stage.

Evaluate transfer on the exact 400-ID EgoNormia correct-video development
panel at 1,024 output tokens with unchanged prompts and frames. Compare
action, justification, joint accuracy, sensibility IoU, and invalid rates
against the saved RLOO starting model using paired source-video-cluster
intervals. A source-task action gain without EgoNormia joint gain is a
negative transfer result. The EgoNormia final source-disjoint partition is
reserved for one confirmatory run only after method selection. The
training corpus lacks source-trajectory IDs, so MUSON validation is not
guaranteed trajectory-disjoint; its class imbalance and navigation-only
scope also limit generalization claims.
