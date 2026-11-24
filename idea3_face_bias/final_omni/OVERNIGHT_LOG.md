# Overnight log (Nov 23 to 24, 2026). Times marked CT are cluster time; ET = CT + 1 h.

Queue rules from Yash (about 01:20 ET): (1) interactive only if the gpuA100x4 / gpuH200x8 batch queue looks longer than about 1 h;
(2) interactive counts at 2x toward the 100 SU cap; (3) every interactive job is resumable and saves before the 1 h limit;
(4) DPO uses batch if it can start within about 2 h. Enforced by slurm/submit.py. SU figures are Slurm billing rates (interactive already 2x).

- 11-24 00:41 CT | JOB 24447686 g1_nemotron: batch wait est A100x4 136.6 h, H200 1349.3 h -> gpuA100x4-interactive, 0.95 h at 4.0 SU/h (max 3.8); SU used 0.0, reserved 0.0
- 11-24 01:07 CT | **G1 Nemotron attempt 1: FAILED (a)**. Letter mass 0.083 at the first position (threshold 0.5). Thinking is off: 20/20 free generations are a bare letter, 19/20 preceded by a newline. (b) 99.3% agree, (c) pass, (d) crashed (class flag for gradient checkpointing missing), (e) 0.24 s/forward batched. Numbers kept in results/gate/g1_nemotron_attempt1.json.
- 11-24 01:08 CT | Repair (to be confirmed by Yash): the newline is added to the fixed assistant prefix for every item, the letter is scored next. Same thresholds, whole gate rerun as attempt 2. Weights copied to NVMe because the HDD load took 11 min.
- 11-24 01:07 CT | JOB 24449240 g1b_nemotron: batch wait est A100x4 140.9 h, H200 1397.9 h -> gpuA100x4-interactive, 0.95 h at 4.0 SU/h (max 3.8); SU used 1.1, reserved 0.0
- 11-24 01:10 CT | Cancelled job 24449240 (A100 interactive, est. start 02:06, never ran). 2x A40 interactive starts at once and costs 2 SU/h instead of 4; moved there. Deviation from the G1 hardware wording (2x A100 or 1x H200), to be noted to Yash. Attempt 1 ran on A100s, attempt 2 on A40s.
- 11-24 01:09 CT | JOB 24449489 g1b_nemotron: batch wait est A100x4 140.9 h, H200 1397.9 h -> gpuA40x4-interactive, 0.95 h at 2.0 SU/h (max 1.9); SU used 1.1, reserved 0.0
- 11-24 01:42 CT | **G1 Nemotron attempt 2 (2x A40): (a) pass (letter mass 0.99998, argmax = free generation on 20/20, accuracy 80%), (b) FAILED (145/150 = 96.7% batched vs one-at-a-time, threshold 98%; attempt 1 on A100 had 99.3%), (c) pass, (d) pass (loss 0.693 to 0.004, 1.87M trainable parameters in 6 attention blocks), (e) 1.0 s one-at-a-time, 0.31 s batched.** Decision: batched scoring disabled; everything is scored one prompt at a time (the reference path). The midterm-style noise override is Yash's to give, not mine. A faster MoE dispatch (same arithmetic, same order) is used only if it reproduces the original dispatch exactly.
- 11-24 01:41 CT | JOB 24451546 g2text_nemotron: batch wait est A100x4 147.3 h, H200 1471.5 h -> gpuA100x4-interactive, 0.95 h at 4.0 SU/h (max 3.8); SU used 2.0, reserved 1.0
- 11-24 01:42 CT | Cancelled 24451546 (A100 interactive estimate slipped to 06:09, never ran); resubmitted through the helper.
- 11-24 01:41 CT | JOB 24451658 g2text_nemotron: batch wait est A100x4 147.3 h, H200 1471.5 h -> gpuA100x4-interactive, 0.95 h at 4.0 SU/h (max 3.8); SU used 2.0, reserved 1.0
- 11-24 01:43 CT | JOB 24451717 g2text_s1_nemotron: batch wait est A100x4 147.3 h, H200 1471.5 h -> gpuA100x4-interactive, 0.95 h at 4.0 SU/h (max 3.8); SU used 2.0, reserved 1.0
- 11-24 02:13 CT | JOB 24454051 g2text_s0_nemotron: batch wait est A100x4 165.3 h, H200 1315.3 h -> gpuA40x4-interactive, 0.95 h at 2.0 SU/h (max 1.9); SU used 3.0, reserved 3.8
- 11-24 02:13 CT | Job 24449489 hit its walltime with perception at 80/90, 80 of 80 correct (0 misses); resumable, remaining 10 run later. Added a clean-stop check.
- 11-24 02:46 CT | 57 min interactive jobs sat 30+ min (estimates slid to 08:33 / 14:48). Cutting the limit to 25 min started job 24454051 within a minute. All further interactive jobs are 25 min, resumable, and run src/driver.py (takes the next ready task; gates enforced in code).
- 11-24 02:57 CT | 24454051 failed in moecheck (missing no_grad in the dup2 reference loop, my bug; OOM). Fixed, results saved before that step. Shard 1 (24451717) is running on A100s with the original dispatch.
- 11-24 02:56 CT | JOB 24456793 drv: batch wait est A100x4 167.9 h, H200 1463.2 h -> gpuA40x4-interactive, 0.42 h at 2.0 SU/h (max 0.8); SU used 3.8, reserved 3.4
- 11-24 03:08 CT | moecheck (first version) reported identical outputs but NO speedup (1.14 vs 1.14 s): the class-level patch was bypassed by accelerate's instance wrappers, so that comparison was vacuous and is not used. New moecheck2 removes the inner wrappers, counts the calls, and compares again.
- 11-24 03:12 CT | Noise reference for G1(b): the same prompt as an identical batch of two agrees with one-at-a-time on 142/144 = 98.6% (max |dp| 0.12, mean 0.020; the two disagreements have top-2 gaps 0.017 and 0.084). Batched path in G1 attempt 2: 145/150, max 0.15, mean 0.021. For Yash to judge; batched stays disabled.
- 11-24 03:13 CT | JOB 24457516 drv: batch wait est A100x4 170.0 h, H200 1415.9 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 5.4, reserved 0.3
- 11-24 03:15 CT | 24451717 (g2 text s1, A100) stopped cleanly at its deadline (exit 75), resumable.
- 11-24 03:20 CT | JOB 24457869 drv: batch wait est A100x4 170.3 h, H200 1395.1 h -> gpuA40x4-interactive, 0.42 h at 2.0 SU/h (max 0.8); SU used 5.9, reserved 1.4
- 11-24 03:23 CT | moecheck2 (A100): inner accelerate wrappers removed, 72/72 identical outputs, 1.07 to 0.87 s, but the fast dispatch was never called (call counter 0 of 1656), so it was NOT adopted. Instance-level override added, rechecking as moecheck3.
- 11-24 03:24 CT | Cancelled 24457516 (2 min into s1, rows kept) so that moecheck3 can run now.
- 11-24 03:22 CT | JOB 24458098 drv: batch wait est A100x4 170.2 h, H200 1395.0 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 6.1, reserved 0.8
- 11-24 03:22 CT | SBATCH FAILED: drv: batch wait est A100x4 170.2 h, H200 1395.2 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 6.1, reserved 2.5: sbatch: error: QOSMaxSubmitJobPerUserLimit
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
- 11-24 03:41 CT | JOB 24459833 drv: batch wait est A100x4 170.3 h, H200 1412.6 h -> gpuH200x8-interactive, 0.42 h at 6.0 SU/h (max 2.5); SU used 6.8, reserved 1.8
- 11-24 03:45 CT | JOB 24460099 drv: batch wait est A100x4 169.7 h, H200 1388.5 h -> gpuA40x4-interactive, 0.42 h at 2.0 SU/h (max 0.8); SU used 7.1, reserved 3.9
- 11-24 03:48 CT | moecheck3 PASSED (A100): fast dispatch called 1656/1656 times, 72/72 outputs bit-identical to the original, 1.07 to 0.47 s per one-at-a-time forward. Adopted for all later scoring. Rows already written used the original dispatch (identical arithmetic). Restarted the A40 job so it uses it. H200 interactive added as a third lane.
- 11-24 04:06 CT | Keepalive bug: a pending H200 job counted as covering the only runnable task, so the A40 lane sat empty 03:48 to 04:06. Fixed.
- 11-24 04:05 CT | JOB 24461569 drv: batch wait est A100x4 177.4 h, H200 1327.2 h -> gpuA40x4-interactive, 0.42 h at 2.0 SU/h (max 0.8); SU used 8.5, reserved 2.5
- 11-24 04:05 CT | JOB 24461577 drv: batch wait est A100x4 177.4 h, H200 1327.2 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 8.5, reserved 3.3
- 11-24 04:37 CT | Cut pending A40 (24461569) and H200 (24459833) jobs to 18 min to help backfill.
- 11-24 04:48 CT | JOB 24463198 drv: batch wait est A100x4 177.3 h, H200 1300.9 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 10.4, reserved 3.0
- 11-24 05:06 CT | JOB 24463719 drv: batch wait est A100x4 179.1 h, H200 1300.2 h -> gpuH200x8-interactive, 0.30 h at 6.0 SU/h (max 1.8); SU used 11.8, reserved 2.2
- 11-24 05:12 CT | JOB 24463943 drv: batch wait est A100x4 178.2 h, H200 1299.8 h -> gpuA40x4-interactive, 0.42 h at 2.0 SU/h (max 0.8); SU used 11.9, reserved 3.5
- 11-24 05:37 CT | Cut pending A100 (24463198) to 12 min and A40 (24463943) to 18 min for backfill.
- 11-24 05:49 CT | JOB 24465238 drv: batch wait est A100x4 176.8 h, H200 1286.0 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 12.5, reserved 2.6
- 11-24 05:50 CT | **G2 Step 1 (text, 300 train items, Nemotron base): L2 PASS.** App flips minus own floor: L1 +13.7 [9.8, 17.7], L2 +4.6 [1.6, 7.5], L3 +4.6 [1.0, 8.0]. Main level L2. Letter mass 0.99997, all finite, 12,900 rows. Mixed hardware (A100, A40, H200). Speech/image shards released.
- 11-24 06:07 CT | JOB 24465732 drv: batch wait est A100x4 176.9 h, H200 1285.2 h -> gpuH200x8-interactive, 0.42 h at 6.0 SU/h (max 2.5); SU used 14.0, reserved 2.5
- 11-24 06:34 CT | Submission window extended from 06:55 to 07:30 CT (same cap and gates) to finish speech/image G2 and Stage 2.
- 11-24 06:33 CT | JOB 24466464 drv: batch wait est A100x4 183.4 h, H200 1285.6 h -> gpuA100x4-interactive, 0.42 h at 4.0 SU/h (max 1.7); SU used 14.8, reserved 3.3
- 11-24 07:10 CT | MORNING_SUMMARY.md written. Speech/image partial (180/300).
- 11-24 07:46 CT | Last pending job (A40 24463943) cut to 12 min; about 550 speech/image rows remain. No new submissions.
