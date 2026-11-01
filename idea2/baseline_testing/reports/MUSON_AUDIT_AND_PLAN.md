# MUSON data audit and candidate action-aligned RL follow-up

The public [MUSON v1.0 release](https://github.com/MUSON-dataset/MUSON/releases/tag/v1.0)
was downloaded to Ocean and verified as a valid ZIP with SHA256
`69eeead852191531f06eb0dd474d322dda9bb104ac7aac91b534384ec1c30b06`.
It has 10,110 RGB images and a JSON annotation file with exactly one
record per image. The [authors' paper](https://arxiv.org/html/2512.22867)
describes these as first-person social navigation scenes with AI-assisted,
human-reviewed perception, prediction, reasoning, action, and explanation
annotations. The downloadable release contains still images, not source
video clips or source-trajectory identifiers. Its GitHub repository does not
declare a license; no images or annotation text will be redistributed here.

The released action labels have six values. Counts are move forward 5,508,
forward right 2,300, forward left 1,564, stop 555, turn right 107, and turn
left 76. This class imbalance makes overall action accuracy alone
uninformative; a model that always chooses forward reaches 54.5%.
The dataset provides a single preferred action per image, even though some
alternatives may be socially acceptable. It therefore cannot directly
supervise EgoNormia's sensibility IoU or its five-way, action-conditioned
justification choice.

For an independent pilot, records are partitioned by a fixed SHA256 hash of
sequential 100-image blocks. This yields 7,710 train, 1,100 validation, and
1,300 reserved test examples. Keeping adjacent numbers together reduces
likely frame-neighbor leakage, but source trajectories are not identified,
so the split cannot be certified as trajectory disjoint. The source data
and all derived images remain under Ocean `data/muson`; no EgoNormia
examples are used in preparation or training.

MUSON is a **secondary** source for the reasoning workstream. R³-FDT remains
the primary video RL training source and supplies the starting RLOO adapter.
Unlike the previous NormLens pilot, MUSON directly asks for an embodied
assistant's next action and provides explanations from an egocentric view.
It is restricted to navigation and single-frame observations, so transfer
to broader first-person social norms is uncertain.

The next engineering gate evaluates the R³-FDT RLOO adapter on 300
deterministically selected MUSON validation images. If the format works,
train a fixed small RLOO continuation with six permuted action choices and
an exact-action reward, balancing sampling across the six action classes to
avoid optimizing the forward majority. Keep the final 1,300-image MUSON test
split and the EgoNormia final partition untouched while developing. Evaluate
on the same 400-ID EgoNormia development panel at 1,024 tokens and compare
to the RLOO starting checkpoint with paired source-video intervals.
If transfer is promising, a matched action-only SFT control and an RL
reward addressing explanation consistency should test whether RL contributes
beyond extra task-specific data.
