# Morning summary, Tue Nov 24 (written 07:10 CT = 08:10 ET)

**Short version.** Nemotron works with thinking off and with steering hooks and LoRA. It hears and reads all cues (100%). G1 has one
open item for you (batched scoring, see below). G2 text passes at L2, but the L2 effect is small (+4.6 points). Speech and image G2 are
half done and queued. No test-split run was made. Nothing was pushed to GitHub (local commits only, branch idea3-final).

## What ran (Nemotron, main model)
| Step | Result |
|---|---|
| G1 attempt 1 | FAILED (a): letter mass 0.083, because the model answers "\nB", a newline before the letter. (d) crashed on a missing class flag |
| G1 attempt 2 | (a) pass after adding the newline to the assistant prefix: letter mass 0.99998, 20/20 generations match, accuracy 80% on 20 val items; (b) FAILED: batched agrees with one-at-a-time on 145/150 = 96.7% (needs 98%); (c) pass; (d) pass: LoRA loss 0.693 to 0.004 in 10 steps |
| Perception check | PASS: 90/90 (text, speech, image each 30/30) |
| G2 text, 300 train items | PASS at L2 (table below) |
| G2 speech and image at L2 | 180 of 300 items done, NOT a gate decision yet (table below) |
| Directions, valence control, F2, sweep, FREEZE, DPO pairs and training, second model | not reached |

## G2 text (flip rates in %, 95% CI clustered by source video, 300 items, base model)
| Level | App flips | Floor | App minus floor | Behavior flips | Acc low / high | Low minus high |
|---|---|---|---|---|---|---|
| L1 | 18.0 [14.3, 21.8] | 4.3 [2.1, 6.9] | **+13.7 [9.8, 17.7]** | 25.8 | 48.6 / 54.4 | -5.9 [-9.4, -2.7] |
| L2 (main) | 10.6 [8.1, 13.3] | 6.0 [3.6, 8.8] | **+4.6 [1.6, 7.5]** | 17.3 | 52.9 / 54.4 | -1.6 [-3.9, 0.7] |
| L3 | 13.6 [10.8, 16.4] | 9.0 [5.9, 12.4] | **+4.6 [1.0, 8.0]** | 16.2 | 51.3 / 54.6 | -3.2 [-5.8, -0.8] |

### G2 speech and image at L2, PARTIAL (180 of 300 items, the first 150 plus 30; do not treat as the gate)
| Channel | App flips | Floor | App minus floor | Behavior flips | Low / high acc |
|---|---|---|---|---|---|
| speech | 5.0 [3.0, 7.3] | 3.9 [1.2, 6.9] | +1.1 [-2.2, 4.3] (no effect so far) | 11.4 | 40.7 / 41.7 |
| image | 13.5 [10.1, 17.2] | 7.2 [3.6, 11.4] | +6.3 [1.9, 10.4] | 20.3 | 43.0 / 45.9 |

On the same 180 items the neutral remark itself costs accuracy: text 53.3%, speech 43.9%, image 46.1% (no remark 54.4%). Perception is perfect,
so the cue is heard and read; the spoken and captioned remark degrades the task more than the typed one. Check this before
reading the speech null as "the model ignores appearance in speech".

Clean accuracy 56.0% [50.3, 61.6]. Midterm (Qwen3-VL) for context, with the bare sentence: app 25.3, floor 2.0, behavior 31.3.
Reading: plain sentences (L1) bias Nemotron clearly; embedding the cue in a description (L2) or a bystander remark (L3) weakens it.

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

## SU used: 14.8 of the 100 cap (from sacct and the job ledger); up to 5.0 more reserved by 3 pending jobs. All on bjuv-delta-gpu.
No job ran on bjyy. One pending job per interactive partition will keep taking G2 and Stage 2 work if it starts after I stop at 07:30 CT.

## Not done / known gaps
- Directions, valence control, F2, steering sweep, FREEZE.md, DPO pairs and training, second model: not reached. Reason: GPU queue, one job per
  partition, 20 to 50 min waits per 25 min job.
- Qwen3-Omni: tokenizer, processor and key mapping checked on CPU only. No G1 for it yet. D2b does not apply (Nemotron failed only the
  batched-path criterion, which is not among its listed causes), but that is your call.
- The gate rows use 300 train items; floors are higher than at midterm because neutral and paraphrase reorder a full sentence.

## Bugs I made overnight, logged
Missing no_grad in moecheck (OOM); a first fast-MoE check that was vacuous (patch bypassed, caught because timing did not change); a call-count
check that caught a second bypass; a keepalive bug that left the A40 lane empty for 17 minutes; cancelling a job that then waited 17 minutes.

## Next (for this session after you reply)
Finish speech/image G2, then directions and the valence control, then the sweep. Before the DPO spend, decide decision 3, and consider that
the L1 effect is three times the L2 effect.
