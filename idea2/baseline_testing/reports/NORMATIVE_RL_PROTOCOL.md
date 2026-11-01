# Exploratory final-project experiment: visual norm reward after R³-FDT RL

The completed R³-FDT RLOO and GRPO pilots improved R³-FDT source-task
accuracy, but only RLOO showed a small, uncertain EgoNormia point gain.
At 1,024 output tokens, GRPO remained below the starting model, while SFT
was the strongest of the four. Raising the response cap beyond 384 reduced
invalid answers without materially changing joint accuracy. These results
motivate changing the **reward target**, rather than the length or RL optimizer.

The independent [NormLens corpus](https://arxiv.org/abs/2310.10418) contains
human judgments of whether an action is wrong, okay, or impossible in a
photographed setting. Its high-agreement subset has 934 examples. This is a
secondary *image* normative source; it is not R³-FDT, EgoNormia, or a video
training set. We use it to test whether adding visually grounded normative
RL after the R³-FDT video RL stage improves transfer. This remains the
reasoning workstream, not a reproduction of the team's NormLens SFT work.

The verified public archive is pinned to SHA256
`b92f190bca09b748f97a94b2fe058908a93fa3e4ca0025d90f324cd0ed08eba7`.
Only unanimous high-agreement labels enter training. Image-disjoint fixed
SHA256 splits are 716 train / 123 validation / 95 test; source images and
annotations stay on Ocean. The held-out test split is reserved until this
pilot is fixed. EgoNormia labels are never inputs to training or checkpoint
selection. The original EgoNormia final partition remains untouched.

Initialize from the frozen R³-FDT RLOO checkpoint at step 100. For each
NormLens train image and proposed action, randomly permute the three
judgment options, sample four completions, and reward a completion by one
only when its final letter matches the unanimous human judgment. Apply the
same one-update-per-mixed-group RLOO policy-gradient objective with sampled
KL to the base model as in the original R³-FDT pilot. The adapter is
language-only rank 8; the Qwen3.5-9B backbone remains NF4 frozen. Fix 25
accepted optimizer updates for the first exploratory continuation, one seed,
and a 192-token training response cap. The initial two-update engineering
pilot attempted 20 examples, accepted one update, and logged five
nonfinite-gradient attempts before the FP16 loss scaler backed off;
the new run initializes the scale at 1024 and still logs any skipped updates.
The reference policy remains the unmodified base model, not the RLOO
initialization, a limitation of this pilot. No human explanations are used as
generated rationale targets.

Evaluate the starting RLOO checkpoint and continuation on the image-disjoint
NormLens validation split. Then run the frozen continuation on the exact same
400-ID EgoNormia correct-video panel at the established 1,024-token cap,
comparing against the saved Base/RLOO/GRPO/SFT results. Report source-task
reward, skipped groups, normative validation accuracy, EgoNormia native
metrics, invalid rates, and paired source-video-cluster intervals. A positive
EgoNormia result on this reused panel is exploratory; a fresh source-video-
disjoint EgoNormia final panel should be scored once after method selection.

This experiment targets the descriptive-to-normative supervision gap. It
does not assume that an image-based right/wrong/impossible task fully matches
EgoNormia's first-person action and justification tasks. Additional controls
are needed before attributing any gain specifically to RL or to visual
grounding: matched normative SFT, text-blind normative RL, and an equal-update
R³-FDT continuation are candidates if the pilot is promising.
