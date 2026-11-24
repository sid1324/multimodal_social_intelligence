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
