# Reasoning workstream: reinforcement learning for video social reasoning

**Measured midterm results on the supplied 400-ID panel.** This is a research
record; the shorter paste-ready section is `reports/MIDTERM_SECTION.md`.

## Initial implementation (target: about one page)

Our research idea is to improve normative decisions in egocentric situations
by reinforcing video-grounded social reasoning on an external training source.
We use R3-FDT, which asks about social events, mental states, and causal
relationships in movie clips. We evaluate transfer on EgoNormia, whose tasks
concern appropriate actions, their justifications, and socially sensible
alternatives. Learning what a character believes or does is not equivalent to
learning what an assistant ought to do; transfer across this distinction is
the central hypothesis.

The starting model is the post-trained, multimodal Qwen3.5-9B checkpoint, pinned
to the revision in our configuration. We quantize supported linear weights to
NF4 and train low-rank adapters in language-model attention projections while
freezing the vision encoder and base parameters. Both the unmodified baseline
and adapted model use the same quantized backbone and numerical preparation.
Eight timestamped frames are sampled uniformly from each clip and passed as
four ordered two-frame segments because the installed Qwen3.5 processor
otherwise produces a video-position mismatch. Audio and
transcripts are excluded in this initial implementation. Dialogue-dependent
questions therefore remain a potential source of missing information.
An engineering smoke test on one V100-32 GB passed real R3 video generation,
finite adapter gradients, an optimizer step, and adapter reload parity; peak
reserved GPU memory was 13.01 GB. This is a feasibility check, not an
evaluation result.

Let x=(V,q,O) denote video evidence, question, and five options; let a* be the
released answer index. The policy pi_theta generates a bounded response y
containing an explanation and a final answer. The verifiable reward is

    r(x,y) = 1[the final answer is valid and equals a*].

This reward validates the answer but does not establish that the explanation
faithfully uses visual evidence. We randomly permute the options and remap a*
during training. For G=4 on-policy samples, define the leave-one-out advantage

    A_i = r_i - (1/(G-1)) sum_{j != i} r_j.

Our initial RLOO policy-gradient loss is

    L = (1/G) sum_i [ -A_i/Lmax * sum_t log pi_theta(y_it | x,y_i,<t)
                      + beta/Lmax * sum_t (exp(d_it)-d_it-1) ],
    d_it = log pi_ref(y_it | x,y_i,<t) - log pi_theta(y_it | x,y_i,<t).

The second term is the sampled reference-KL surrogate used in this
implementation; it is not a verified reasoning-quality reward. The reference
is the frozen starting policy. We take one update per group and generate at
temperature one without top-k/top-p truncation. Lmax is a fixed response cap,
so the denominator does not vary with realized response length. Groups with
identical correctness rewards are skipped and counted. This filtering and
the binary reward's limited signal are explicit limitations.

R3 partitions are disjoint by movie wherever upstream metadata provides movie
identity. EgoNormia supplies no training targets. We reserve source-video
groups for final evaluation before inspecting midterm errors, and use a fixed
team-supplied 400-ID midterm panel for the present analysis. Our EgoNormia evaluator preserves the
official prompts, conditions justification on the predicted action, and uses
only preceding video. It reports joint action/justification accuracy, separate
accuracies, and mean sensibility IoU, including failures in the denominator.

## Results and discussion (target: 0.75–1 page)

The team-supplied midterm sample contains 400 EgoNormia IDs from 348 source
videos, including 46 Verified examples. All 400 have official repository
annotations; 24 are absent from the HF annotation parquet, so their preceding
clips were checked separately. All 400 preceding clips decoded successfully.
The released correct-action indices occur
108, 104, 97, 87 and 4
times for positions 1–5 respectively. This unequal distribution means chance
and position-prior comparisons should use the fixed sample, not assume every
position occurs equally often. Norm categories overlap, so category counts do
not sum to 400. Video acquisition and inference coverage must be recorded
separately; these are annotation counts, not model results.

The initial R3 media pool yielded 96/100 videos: 84 train videos (1,262
questions), seven validation videos (86 questions), and five test videos
(79 questions). The small held-out video counts limit precision even if the
question counts look larger. Three selected upstream videos were unavailable
and one returned HTTP 403. Results below must be attributed to this accessible
pool rather than all 41,529 released questions.

The frozen base-model evaluation used Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, seed 777, and the same NF4
configuration used for later adapters. Correct-video inference now covers all
400 supplied panel IDs with identical settings for base, RLOO, and SFT; no
inference exception occurred. All 400 clips were available and decoded. The
original time-limited pass covered the first 140 IDs from 133 source videos
with all three input conditions. Those 140-ID results remain useful controls,
but the 400-ID correct-video comparison is the primary midterm transfer test.

| Model / correct video | n/panel | Action | Justification | Joint | Sensibility IoU | Invalid examples |
|---|---:|---:|---:|---:|---:|---:|
| Frozen base | 400/400 | 44.8 | 36.2 | 23.8 | 46.5 | 167 |
| RLOO adapter | 400/400 | 45.8 | 38.2 | 26.2 | 46.3 | 165 |
| Answer-only SFT | 400/400 | 47.8 | 35.5 | 28.0 | 46.8 | 129 |

On all 400 paired examples, RLOO-minus-base joint accuracy is +2.5 percentage
points (source-video-cluster bootstrap 95% interval −0.8 to +5.7), and
SFT-minus-base is +4.3 points (+0.5 to +8.0), while SFT-minus-RLOO is
+1.8 points (−2.5 to +5.9). RLOO's interval versus base includes zero;
the SFT control's interval does not. These intervals reflect example sampling
on one model seed, not variation across training seeds. The RLOO result is an
uncertain positive point estimate, while SFT improved measured joint accuracy
on this panel. No RL-specific transfer advantage is demonstrated. The 46
panel examples in EgoNormia-Verified have joint scores of 28.3%, 34.8%, and
30.4% for base, RLOO, and SFT respectively; this panel intersection is not
the official 200-example Verified evaluation.

The correct-action norm tags overlap. The following are descriptive joint
scores on the 400-ID panel, with no category-specific significance claim:

| Norm tag | n | Base | RLOO | SFT |
|---|---:|---:|---:|---:|
| Safety | 122 | 25.4 | 26.2 | 27.9 |
| Privacy | 34 | 20.6 | 29.4 | 23.5 |
| Proxemics | 128 | 25.8 | 31.2 | 29.7 |
| Politeness | 136 | 14.7 | 19.1 | 22.1 |
| Cooperation | 204 | 24.5 | 26.5 | 26.0 |
| Coordination/Proactivity | 86 | 30.2 | 29.1 | 24.4 |
| Communication/Legibility | 108 | 14.8 | 20.4 | 24.1 |

At the 192-token cap, 122 base, 129 RLOO, and 111 SFT justification
responses reach the limit. The stricter final-line validity rule reduces
native-style joint-correct counts from 95 to 83 (base), 105 to 89 (RLOO),
and 112 to 103 (SFT). This is a substantial formatting and generation-length
confound in all models, especially the two reasoning-output policies. It does
not change the native-style primary score, but it limits interpretation of
small joint differences as social reasoning gains.

The next table is limited to the original 140-ID prefix and includes the
no-video and different-video probes. It should not be combined with the
400-ID table to infer a full-panel visual effect.

| Model / condition | n/panel | Action | Justification | Joint | Sensibility IoU | Invalid examples |
|---|---:|---:|---:|---:|---:|---:|
| Base / correct video | 140/400 | 47.1 | 40.7 | 28.6 | 46.3 | 56 |
| Base / no video | 140/400 | 20.0 | 20.0 | 10.0 | 39.1 | 116 |
| Base / mismatched video | 140/400 | 18.6 | 13.6 | 7.1 | 23.6 | 65 |
| RLOO / correct video | 140/400 | 47.1 | 44.3 | 31.4 | 46.3 | 57 |
| RLOO / no video | 140/400 | 22.1 | 17.9 | 9.3 | 39.3 | 113 |
| RLOO / mismatched video | 140/400 | 17.1 | 12.1 | 7.1 | 22.3 | 54 |
| SFT control / correct video | 140/400 | 49.3 | 39.3 | 32.1 | 45.2 | 42 |

Scores are percentages. The correct-video joint source-cluster bootstrap 95%
interval is 21.1–36.2%; the paired joint differences are −18.6 points for
no video (interval −26.6 to −10.2) and −21.4 points for mismatched video
(interval −29.1 to −14.5). These intervals describe sampling uncertainty on
one fixed model run, not seed-to-seed or training uncertainty. The no-video
condition also changes the introductory wording, while the mismatched-video
condition preserves it; neither perturbation alone proves causal visual
grounding. The 21 attempted examples belonging to the published Verified
subset have 38.1% correct-video joint accuracy, but this is not an official
Verified evaluation.

Within the 140-ID prefix, the paired RLOO-minus-base difference on correct video is +2.9 joint
points (95% source-cluster bootstrap interval −2.2 to +8.5). Action accuracy
is unchanged; justification is +3.6 points (interval −3.6 to +10.5), and
sensibility IoU changes by approximately zero. The gain is a point estimate,
not a statistically resolved improvement. The RLOO model still falls on
no-video and mismatched-video controls; its paired joint drops relative to
its own correct-video condition are 22.1 and 24.3 points.
The answer-only SFT control scores 32.1% joint on the same 140 correct-video
items: +3.6 points versus base (interval −2.9 to +10.1) and +0.7 versus
RLOO (interval −5.8 to +7.5). Neither adaptation has a resolved gain on this
shorter prefix; the full 400-ID result above supersedes that conclusion.
SFT uses the same R3 training manifest and 100 nominal updates, but its
answer-only cross-entropy objective, examples visited, and compute differ
from RLOO: it used 113 distinct questions from 63 videos/60 movie groups in
about 5.3 minutes, with 13 early nonfinite-gradient skips.
SFT has fewer invalid correct-video outputs (42/140) than either base
(56/140) or RLOO (57/140); strict-format joint correctness is 42/140 for
SFT versus 38/140 for RLOO. This supports treating response format as a
separate mechanism in the control comparison.

For completeness, correct-video joint accuracy by overlapping norm tag on the
140-ID prefix is below. These smaller dependent groups do not support
category-specific claims; the 400-ID table above has broader coverage.

| Norm tag | n | Base | RLOO |
|---|---:|---:|---:|
| Safety | 46 | 26.1 | 32.6 |
| Privacy | 8 | 25.0 | 12.5 |
| Proxemics | 42 | 35.7 | 40.5 |
| Politeness | 47 | 23.4 | 27.7 |
| Cooperation | 70 | 24.3 | 28.6 |
| Coordination/Proactivity | 30 | 36.7 | 36.7 |
| Communication/Legibility | 35 | 14.3 | 22.9 |

Output formatting is a substantial confound. On correct video, 16 action and
42 justification responses fail our stricter final-line validity check; 15
and 45 respectively hit the 192-token cap. Native-style last-integer parsing
still credits five joint-correct examples with invalid output. Requiring both
answers to satisfy the strict final-line format reduces joint correctness
from 40/140 to 35/140. This does not change the primary native-style score;
it reveals how response formatting affects its interpretation. The RLOO model
has 15 invalid action and 42 invalid justification outputs; strict-format
joint correctness is 38/140 versus native-style 44/140. Thus the adapter did
not materially resolve output-format failures. No inference exceptions
occurred in either model's 140 correct-video attempts.

An exploratory model-by-response-cap diagnostic used the first 40 fixed panel
IDs, selected by supplied order rather than correctness:

| Model | 192-token joint | 384-token joint | Invalid examples, 192→384 |
|---|---:|---:|---:|
| Base | 7/40 (17.5%) | 12/40 (30.0%) | 19→4 |
| RLOO | 10/40 (25.0%) | 13/40 (32.5%) | 18→5 |

The base-model cap change is +12.5 paired points (source-cluster interval
+2.5 to +23.1); the RLOO cap change is +7.5 points (interval 0 to +15.4).
At 384 tokens, RLOO exceeds the base by only one item (+2.5 points; interval
0 to +7.7). On this small slice, completion length has a larger effect than
the adapter point difference. This does not replace the 400-item fixed-cap
comparison and is exploratory because it was motivated by observed errors.

We then extended the 384-token diagnostic to the exact same 400 IDs for all
three models, preserving frames and prompts. The original first-40 cap runs
were merged with audited remaining-360 runs for base and RLOO; SFT was run
on all 400. All jobs completed without inference exceptions.

| Model | 192-token joint | 384-token joint | Paired cap gain, 95% source-group interval | Invalid examples, 192→384 |
|---|---:|---:|---:|---:|
| Base | 23.8% | 30.8% | +7.0 points (+4.2 to +10.0) | 167→48 |
| RLOO | 26.2% | 33.5% | +7.3 (+4.7 to +10.0) | 165→42 |
| SFT | 28.0% | 35.0% | +7.0 (+4.6 to +9.4) | 129→41 |

At 384 tokens, RLOO minus base is +2.8 joint points (paired cluster 95%
interval −0.5 to +6.1), SFT minus base is +4.3 (+0.8 to +7.7), and SFT
minus RLOO is +1.5 (−2.6 to +5.6). Thus the model ordering and unresolved
RL-specific advantage persist at the longer cap. The 7-point response-cap
effect is larger than the RLOO point gain and remains a major inference
limitation. Because the cap extension was motivated by midterm errors, we
label it a diagnostic rather than silently changing the primary protocol.
The paired 192→384 transitions support a truncation explanation: base gains
32 joint-correct examples and loses four, RLOO gains 30 and loses one, and
SFT gains 28 and loses none. By comparison, at the original 192-token cap,
RLOO fixes 27 base failures but regresses on 17 base successes; 278 remain
wrong for both. The +10 net item gain therefore hides substantial turnover,
not a uniform correction of the same social-norm failure mode.

On the accessible movie-disjoint R3 partitions, the unmodified model scores
62.5% on 40/86 validation questions and 68.3% on 60/79 test questions with
video. The same 60 test questions score 61.7% without video; this modest
in-domain visual advantage leaves room for language/option shortcuts. The
RLOO run completed 100 optimizer updates in 84.1 minutes on one V100. It
sampled 156 distinct R3 training questions from 73 videos and 70 movie
groups, generating 624 completions. Mean sampled answer correctness was
58.7%, 49 groups had uniform rewards and were skipped, seven early groups
had nonfinite gradients during loss-scale adjustment, and 26 completions hit
the 192-token cap. Mean sampled KL on update candidates was 0.0108, with a
maximum of 0.117. The saved adapter is checkpoint 100; the trace and figure
are in `reports/generated/rloo_training/`. These are optimization diagnostics,
not evidence of EgoNormia transfer. On the same R3 held-out questions, the
adapted model reaches 70.0% on 40 validation questions and 80.0% on 60 test
questions with video, versus 62.5% and 68.3% for the base model. The test
video gain is +11.7 points (paired source-group bootstrap interval +2.7 to
+21.3), while no-video test accuracy remains 61.7% for both models. This is
consistent with an in-domain benefit from video-conditioned R3 training, but
the test slice contains only five source-video groups and 60 questions. Thus,
RL improved the measured R3 test slice but has only an uncertain small
EgoNormia point gain on 400 examples. This is consistent with a
descriptive-to-normative transfer gap; it does not establish that gap as the
unique cause.
The SFT control reaches 85.0% on the same 40 R3 validation questions and
93.3% on the same 60 R3 test questions. Its larger in-domain gain accompanies
a smaller but measurable +4.3-point EgoNormia joint gain on the fixed panel;
the gap in gain size still limits claims of broad normative transfer.

Among the 140 correct-video items, 40 have both action and justification
correct, 26 action only, 17 justification only, and 57 neither. This makes
action/justification consistency a more informative diagnostic than either
component alone. In the food-stand clip
`82e96128-1242-4e57-8a3e-9ea222df022c_2985-80`, the model's prose
identifies ordering food as the appropriate action but outputs option 1,
offering a napkin, instead of the released option 3. The video supports its
scene description; the observed error occurs at final option selection. In
workshop clip `e3f37c38-bf24-481f-b1c6-8d472fab4290_1679-92`, its prose
advocates the released safety action, offering eye and ear protection, but
reaches the generation cap before printing an integer; the subsequent
justification is conditioned on no selected action. This is a format and
conditioning failure, not a clean failure to recognize the safety norm.

The 6.37-second sidewalk clip
`9d965c16-0014-45e5-b37b-50a786971470_8107-86` shows a group walking
toward a street, while the released best action refers to joining a group at
a restaurant table. The model chooses to follow a supposed tour guide; the
clip does not establish a tour guide either. This is a possible label-context
ambiguity in a preceding-video-only protocol, so it should not be used as an
unambiguous model failure. A successful case is
`f0041c6c-06b2-4c3b-b8e6-934305b4226e_6269-35`: the model selects
continuing to walk with a nearby group and the matching group-cohesion
justification. The action is supported by frames; the conversation part of
its wording is not independently audible. Frame sheets and detailed notes
are in `reports/generated/error_review/` and `reports/CASE_NOTES.md`.

A paired regression shows that correct action and justification need to be
checked together. In salon clip
`c34effc7-6198-4c85-80c7-c8be85b77dd3_2128-22`, frames show a stylist
working on a client's hair while the client looks at a phone. Base selects
the released action of continuing product application without discussing the
phone and its privacy justification. RLOO retains the correct action but
selects a different justification about proactively accommodating the
client's activities; its prose describes engaging with the phone. This is a
specific answer-consistency regression, not proof of a category-wide effect.

## Updated research idea (target: about half a page)

RLOO improves the small R3 test slice while the EgoNormia joint gain remains
uncertain. Correct-video baselines outperform mismatched video, but the model
still makes option-mapping and temporal-stage errors, and many responses hit
the output cap. The next research experiment will target **cross-modal
reliance** rather than only more R3 correctness updates. On R3 training
questions only, compute the frozen model's normalized five-option probability
of the released answer from the question and options *without* video,
`p_blind(a* | q,O)`. For each video-grounded RLOO group, weight the policy
gradient by `w(x)=min(w_max, 1+lambda[1-p_blind(a* | q,O)])`, leaving the KL
penalty unweighted. This prioritizes questions less recoverable from text
alone; it does not intentionally make the no-video policy worse. Compare
weighted RLOO with the present unweighted run and SFT control at matched
R3 question coverage, tracking correct-video, no-video, and mismatched-video
EgoNormia results on a fresh held-out source-video panel. Also stratify by
the frozen model's video difficulty so the weighting is not mistaken for a
generic hard-example effect. No EgoNormia labels enter optimization.

The measured cap ablation means future model comparisons must also hold a
long-enough output cap or constrained final-answer extraction fixed across
models. This is an evaluation control, not the proposed multimodal research
contribution. The shampoo-station case motivates a second, more costly extension: add a
verified temporal-evidence reward to a manually annotated R3 training subset.
If a response cites a frame interval supporting a causal/mental-state answer,
let `r_total=r_answer+mu*r_evidence`, where `r_evidence` is checked against
human-marked intervals, not by plausibility of the model's own explanation.
R3's released five-choice annotations alone do not supply those intervals;
this extension depends on making and auditing them.

## Sources

R3-FDT: https://huggingface.co/datasets/nlxpku/R3-FDT

Read the Room: https://openreview.net/forum?id=TJilJnZjpw

EgoNormia: https://arxiv.org/abs/2502.20490 and
https://github.com/Open-Social-World/EgoNormia

Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B

RLOO background: Ahmadian et al., Back to Basics: Revisiting REINFORCE Style
Optimization for Learning from Human Feedback in LLMs, arXiv:2402.14740.
Our exact implementation and regularizer are specified above.
