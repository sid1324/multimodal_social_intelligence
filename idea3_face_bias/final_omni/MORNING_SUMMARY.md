# Morning summary, Tue Nov 24 (written 08:10 ET, speech/image gate final 09:55 ET, directions added 13:20 ET)

**Short version.** Nemotron works with thinking off and with steering hooks and LoRA. It hears and reads all cues (100%). G1 has one
open item for you (batched scoring, see below). G2 text passes at L2, but the L2 effect is small (+4.6 points). G2 is complete on all 300 items: image passes, speech does not. No test-split run was made. Nothing was pushed to GitHub (local commits only, branch idea3-final).

## What ran (Nemotron, main model)
| Step | Result |
|---|---|
| G1 attempt 1 | FAILED (a): letter mass 0.083, because the model answers "\nB", a newline before the letter. (d) crashed on a missing class flag |
| G1 attempt 2 | (a) pass after adding the newline to the assistant prefix: letter mass 0.99998, 20/20 generations match, accuracy 80% on 20 val items; (b) FAILED: batched agrees with one-at-a-time on 145/150 = 96.7% (needs 98%); (c) pass; (d) pass: LoRA loss 0.693 to 0.004 in 10 steps |
| Perception check | PASS: 90/90 (text, speech, image each 30/30) |
| G2 text, 300 train items | PASS at L2 (table below) |
| G2 speech and image at L2, 300 items | image PASS, speech FAIL (table below) |
| Directions and valence control (CPU) | done (below); F2 is a draft |
| Sweep, FREEZE, DPO pairs and training, second model | not reached |

## G2 text (flip rates in %, 95% CI clustered by source video, 300 items, base model)
| Level | App flips | Floor | App minus floor | Behavior flips | Acc low / high | Low minus high |
|---|---|---|---|---|---|---|
| L1 | 18.0 [14.3, 21.8] | 4.3 [2.1, 6.9] | **+13.7 [9.8, 17.7]** | 25.8 | 48.6 / 54.4 | -5.9 [-9.4, -2.7] |
| L2 (main) | 10.6 [8.1, 13.3] | 6.0 [3.6, 8.8] | **+4.6 [1.6, 7.5]** | 17.3 | 52.9 / 54.4 | -1.6 [-3.9, 0.7] |
| L3 | 13.6 [10.8, 16.4] | 9.0 [5.9, 12.4] | **+4.6 [1.0, 8.0]** | 16.2 | 51.3 / 54.6 | -3.2 [-5.8, -0.8] |

### G2 speech and image at L2 (300 items, final)
| Channel | App flips | Floor | App minus floor | Behavior flips | Low / high acc |
|---|---|---|---|---|---|
| speech | 5.4 [3.8, 7.3] | 5.3 [3.0, 8.0] | +0.1 [-2.9, 2.9] (FAIL) | 11.5 [8.5, 14.7] | 43.1 / 44.6 |
| image | 13.4 [10.7, 16.3] | 8.0 [5.1, 11.3] | **+5.4 [1.7, 9.0]** (PASS) | 21.2 [17.4, 25.1] | 45.4 / 49.3 |

On the same 300 items the neutral remark itself costs accuracy: text 55.3%, speech 46.3%, image 47.7% (no remark 56.0%). Perception is perfect,
so the cue is heard and read; the spoken and captioned remark degrades the task more than the typed one. Check this before
reading the speech null as "the model ignores appearance in speech".

Clean accuracy 56.0% [50.3, 61.6]. Midterm (Qwen3-VL) for context, with the bare sentence: app 25.3, floor 2.0, behavior 31.3.
Reading: plain sentences (L1) bias Nemotron clearly; embedding the cue in a description (L2) or a bystander remark (L3) weakens it.

## Directions at L2 (train split, train wordings, 300 items; cosine averaged over late layers 29 to 49 of 52)
| Comparison | Cosine |
|---|---|
| appearance, text vs speech / text vs image / speech vs image | 0.77 / 0.89 / 0.79 |
| appearance vs behavior, text / speech / image | 0.82 / 0.82 / 0.82 |
| appearance text, L1 vs L2 / L1 vs L3 / L2 vs L3 | 0.92 / 0.94 / 0.90 |
| valence control (weather) vs appearance / behavior, text | 0.36 / 0.34 |

Split-half stability is at least 0.92 for every direction. Mean length of the appearance direction: text 13.5, image 18.2, speech 5.9 (behavior: 19.2, 25.9, 9.3).
Reading: the appearance direction is shared across channels, and speech points the same way but is under half as long as text.
That fits weaker use of the spoken cue, but does not prove it. Cautions: the three channels carry the same wordings, so part of the
overlap can be the words themselves; the valence control uses short weather sentences, so it is not matched in length or form to the
L2 cues; appearance and behavior overlap at 0.82, so steering appearance out may also remove behavior (the sweep must check this).

## Decisions for you
1. **G1(b).** Batched scoring is off; everything uses one prompt at a time, so the failed check does not affect results. Noise reference: the
   same prompt as a batch of two agrees with single runs on 98.6% of 144 (max |dp| 0.12), so the batched gap (96.7%, max 0.15) is close to
   bf16 rounding noise, but your 98% rule was not met. Keep one-at-a-time (slower) or accept batched as at midterm?
2. **Answer prefix.** Is the fixed newline prefix OK? It only moves the scoring position; free generation agrees on 20/20.
3. **Main level.** L2 passes your rule but the effect is small (+4.6), and low-vs-high accuracy is not distinguishable from 0. Keep L2, or make L1
   the main level (strong, +13.7)? Expect few appearance DPO pairs and a small possible reduction at L2.
4. **Channel framing and constitution.** Confirm the remark wording. The constitution arm is off until you approve the added sentence.
5. **Hardware.** The A40 and A100 and H200 were all used. Items never straddle two jobs, so each cue pair is scored on one GPU type. Attempt 2 of
   G1 ran on A40s (your G1 text says A100 or H200).
6. **Budget and queue.** Interactive jobs are the only fast lane: batch estimates were 136 to 180 h on A100x4 and over 1,200 h on H200. DPO is
   long; without batch access it would need many chained 25 min jobs.

## SU used: 17.1 of the 100 cap (sacct). No job is pending or running. All on bjuv-delta-gpu.
No job ran on bjyy. I stopped submitting at 08:30 ET.

## Not done / known gaps
- Steering sweep, FREEZE.md, DPO pairs and training, second model: not reached. Reason: GPU queue, one job per
  partition, 20 to 50 min waits per 25 min job.
- Qwen3-Omni: tokenizer, processor and key mapping checked on CPU only. No G1 for it yet. D2b does not apply (Nemotron failed only the
  batched-path criterion, which is not among its listed causes), but that is your call.
- The gate rows use 300 train items; floors are higher than at midterm because neutral and paraphrase reorder a full sentence.

## Bugs I made overnight, logged
Missing no_grad in moecheck (OOM); a first fast-MoE check that was vacuous (patch bypassed, caught because timing did not change); a call-count
check that caught a second bypass; a keepalive bug that left the A40 lane empty for 17 minutes; cancelling a job that then waited 17 minutes;
the first directions run stalled on shared-disk reads for 3 h 20 min (found at 13:12 ET; local copy took 49 s).

## Next (for this session after you reply)
Run the steering sweep and freeze. Before the DPO spend, decide decision 3, and consider that
the L1 effect is three times the L2 effect.
