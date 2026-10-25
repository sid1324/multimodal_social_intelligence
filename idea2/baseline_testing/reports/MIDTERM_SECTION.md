# Reasoning idea: video-grounded reinforcement learning for social norms

## Initial implementation

Our idea tests whether reinforcement learning (RL) on descriptive social
reasoning from video transfers to an embodied assistant's normative decisions.
R³-FDT asks what people believe, intend, or do in movie clips and provides a
correct answer among five choices. EgoNormia instead asks what an assistant
ought to do next in a first-person scene, why, and which other actions remain
sensible. This descriptive-to-normative difference is the research question,
not an assumption that the datasets provide equivalent supervision.

We start from the open-weight Qwen3.5-9B vision-language model. Its pinned
checkpoint is loaded with NF4 4-bit quantization and FP16 computation on one
32 GB V100. The vision backbone and base weights are frozen; rank-8 LoRA
adapters update language attention projections. Eight uniformly sampled,
timestamped frames represent each clip, with aspect ratio retained and a
resolution cap. The installed video processor requires four ordered two-frame
segments for these eight frames to align video tokens with position-grid rows.
Audio and transcripts are absent. A real-video training smoke test passed on
one V100 at 13.0 GB peak reserved memory. Both base and adapted models use
the same quantized backbone, frames, prompts, and decoding settings.

Let $x=(V,q,O)$ be the video, question, and five options in R³-FDT; $a^*$
is the released correct index. We randomly permute options during training
and remap $a^*$. For generated response $y$, the verifiable reward is
$r(x,y)=\mathbf{1}[\mathrm{final}(y)=a^*]$. It scores the final answer,
not the faithfulness of any explanation. With four on-policy rollouts, we use
the leave-one-out advantage
$A_i=r_i-\frac{1}{3}\sum_{j\ne i}r_j$ and minimize

\[
L=\frac{1}{4}\sum_i\left[-\frac{A_i}{L_{\max}}\sum_t\log\pi_\theta(y_{it}\mid x,y_{i,<t})
+\frac{\beta}{L_{\max}}\sum_t(\exp d_{it}-d_{it}-1)\right],\quad
d_{it}=\log\pi_{\rm ref}(y_{it})-\log\pi_\theta(y_{it}).
\]

Here $\beta=0.02$, $L_{\max}=192$, and the reference is the frozen starting
policy. Uniform-reward groups are logged and skipped. The initial pilot uses
100 optimizer updates, four sampled responses per group, and no replay. An
answer-only supervised fine-tuning (SFT) adapter on the same R³-FDT training
manifest is a control for generic task adaptation, not a compute-matched RL
comparison.

We use accessible R³-FDT media only: 1,262 training questions from 84 videos,
86 validation questions from seven videos, and 79 test questions from five
videos, separated by movie. Four of the initial 100 selected upstream videos
were unavailable. R³-FDT annotations are CC BY-NC 4.0; the original movie
rights still apply. The team supplied a fixed 400-ID EgoNormia midterm panel;
all 400 preceding clips were validated. EgoNormia provides no training labels.
We reserve source-video-disjoint examples for final evaluation. This midterm
uses only each item's preceding clip, the benchmark's action and
predicted-action-conditioned justification prompts, and native action,
justification, joint, and sensibility-IoU metrics. A no-video prompt and a
different-source-video control probe reliance on visual context.

## Results and discussion

The correct-video evaluation covers all 400 supplied EgoNormia IDs from 348
source videos, including 46 members of EgoNormia-Verified. The earlier
three-condition pass covers the first 140 IDs from 133 source videos; it
provides the no-video and different-video controls below. Scores are
percentages; invalid outputs remain among attempted examples. These are
midterm-panel results, not full-benchmark or official Verified scores.

| Model / condition | Action | Justification | Joint | Sensibility IoU | Invalid examples |
|---|---:|---:|---:|---:|---:|
| Base / video, 400 | 44.8 | 36.2 | 23.8 | 46.5 | 167/400 |
| RLOO / video, 400 | 45.8 | 38.2 | 26.2 | 46.3 | 165/400 |
| SFT / video, 400 | 47.8 | 35.5 | 28.0 | 46.8 | 129/400 |
| Base / video, first 140 | 47.1 | 40.7 | 28.6 | 46.3 | 56/140 |
| Base / no video | 20.0 | 20.0 | 10.0 | 39.1 | 116/140 |
| Base / mismatched video | 18.6 | 13.6 | 7.1 | 23.6 | 65/140 |
| RLOO / video, first 140 | 47.1 | 44.3 | 31.4 | 46.3 | 57/140 |

Across 400 paired IDs, RLOO's joint difference from base is +2.5 points,
with a source-video-cluster bootstrap 95% interval of −0.8 to +5.7;
SFT's is +4.3 points (+0.5 to +8.0). SFT minus RLOO is +1.8 points
(−2.5 to +5.9). The single-run interval for RLOO versus base includes zero;
SFT's does not. These data do not support an RL-specific advantage. The
intervals address example sampling, not training-seed variation. RLOO took
84.1 minutes for 100 updates on 156 distinct R³-FDT questions. On the same
60 held-out
R³-FDT test questions from five source videos, video accuracy is 68.3% base,
80.0% RLOO, and 93.3% SFT. The base and RLOO no-video test scores are both
61.7%. Thus the measured in-domain gains are clearer than normative transfer,
though the test pool is small. On the 400 panel, base joint accuracy is 14.8%
for Communication/Legibility and 14.7% for Politeness, compared with 25.4%
for Safety; the overlapping tags and unequal counts limit category claims.
Detailed counts are in `reports/MIDTERM_DRAFT.md`.

Failure analysis separates social errors from response-format errors. At the
192-token cap, 122/400 base and 129/400 RLOO justification responses reach
the limit. Requiring the final-line format reduces joint-correct counts from
95 to 83 for base and from 105 to 89 for RLOO. In a food-stand clip, the
base model's prose identifies ordering food as the right
action but its final number selects offering a napkin. At a shampoo basin, it
selects applying shampoo while its justification supports rinsing, the
released next action; the visible sprayer suggests a temporal-stage error.
In a salon privacy clip, RLOO retains the released action of applying product
without discussing the client's phone but changes a correct privacy
justification to one about engaging with the phone, a paired regression in
action/justification consistency.
In a workshop clip, it explains why eye and ear protection is appropriate
but reaches the 192-token limit before printing an answer. A sidewalk clip's
released restaurant-table action is not visible in the permitted preceding
footage, so we record possible label-context ambiguity rather than counting
it as an unambiguous perceptual failure. Detailed frames and notes are saved
without redistributing source videos.

An exploratory 384-token rerun covers all 400 correct-video items with the
same frames and prompts. Joint accuracy becomes 30.8% base, 33.5% RLOO,
and 35.0% SFT, versus 23.8%, 26.2%, and 28.0% at 192 tokens. The paired
cap gains are +7.0, +7.3, and +7.0 points; invalid-example counts fall
from 167→48, 165→42, and 129→41. At 384 tokens, RLOO minus base is
+2.8 points (95% interval −0.5 to +6.1), while SFT minus base is +4.3
(+0.8 to +7.7). The ranking persists, but completion length is a larger
measured factor than the RLOO point gain. This cap diagnostic was motivated
by observed failures and remains separate from the primary fixed-cap result.

## Updated research idea

The R³-FDT gain with unresolved EgoNormia transfer motivates an RL objective
that emphasizes *cross-modal reliance*, not just additional question-answer
practice. For each R³-FDT **training** item, compute the frozen base model's
normalized probability $p_{\rm blind}(a^*\mid q,O)$ of its correct answer
without video. Weight the video-grounded RLOO policy-gradient term by
$w(x)=\min(w_{\max},1+\lambda[1-p_{\rm blind}(a^*\mid q,O)])$, leaving the
reference-KL term unchanged. This prioritizes items that are less answerable
from text alone without training the no-video policy to fail. Compare this
method with unweighted RLOO and SFT at matched R³-FDT coverage, including
video, no-video, and mismatched-video evaluation on a fresh source-video
holdout. Stratify by original video difficulty to distinguish visual need
from generic hard-example weighting. No EgoNormia answer enters training.

The shampoo-station error also motivates a later temporal-evidence reward,
$r_{\rm answer}+\mu r_{\rm evidence}$, but it requires independently checked
frame intervals on R³-FDT training clips; the released answer indices alone
do not provide that supervision. Future evaluations should hold a sufficiently
long response cap fixed across models so answer truncation is not mistaken
for social-norm failure.
