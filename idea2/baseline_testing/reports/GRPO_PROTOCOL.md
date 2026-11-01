# Post-midterm GRPO follow-up (2026-10-28)

This is an exploratory algorithm comparison requested after the frozen midterm
RLOO, SFT, and 400-ID EgoNormia results were available. The existing midterm
protocol and results remain unchanged. The fixed 400-ID EgoNormia panel is
reused only for evaluation, so this is **not** a new blind holdout. No
EgoNormia label enters training.

Model, R³-FDT accessible training manifest, eight-frame video preprocessing,
five-choice correctness reward, option permutation, NF4/FP16 backbone, rank-8
language LoRA, optimizer learning rate, reference model, and one V100-32 GB
match the RLOO pilot. The [DeepSeekMath GRPO paper](https://arxiv.org/abs/2402.03300)
motivates group-standardized advantages and a clipped old-policy ratio. This
implementation is a small video R³-FDT adaptation, not a reproduction of that
paper's large-scale training.

For each question, draw four completions from the current policy and score
binary correct-final-answer rewards. For a mixed-reward group, standardize
rewards by the **population** group standard deviation to obtain one
advantage per completion. Snapshot token log probabilities under the policy
before optimization. Reuse the group for two policy epochs with token-level
importance ratios clipped to [0.8, 1.2] and the same sampled reference KL
penalty (coefficient 0.02). The objective averages over generated tokens
within each completion and then over the four completions. Groups with
uniform reward are logged and skipped. Each epoch has its own gradient step;
nonfinite-gradient epochs are logged and skipped. There is no critic or replay
buffer. The reward checks only final-answer correctness, not rationale
faithfulness or normative action quality.

The requested main run is 100 optimized mixed-reward groups, up to two
accepted optimizer updates per group. This aims for comparable *successful
group count* with RLOO, but uses more optimizer updates and computation; it
is **not** a compute-matched algorithm comparison. Report actual visited
questions/videos, mixed/uniform groups, accepted updates, KL, clipping,
completion caps, runtime, and checkpoint. Evaluate on the same 60
movie-disjoint R³-FDT test questions (video and no-video) and the same 400
EgoNormia correct-video items at 192 tokens. Compare native action,
justification, joint accuracy, and sensibility IoU with paired source-video
cluster intervals. Any 384-token repeat is a separately labeled diagnostic.
