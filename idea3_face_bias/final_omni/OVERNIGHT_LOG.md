# Overnight log (Nov 23 to 24, 2026). Times marked CT are cluster time; ET = CT + 1 h.

Queue rules from Yash (about 01:20 ET): (1) interactive only if the gpuA100x4 / gpuH200x8 batch queue looks longer than about 1 h;
(2) interactive counts at 2x toward the 100 SU cap; (3) every interactive job is resumable and saves before the 1 h limit;
(4) DPO uses batch if it can start within about 2 h. Enforced by slurm/submit.py. SU figures are Slurm billing rates (interactive already 2x).

- 11-24 00:41 CT | JOB 24447686 g1_nemotron: batch wait est A100x4 136.6 h, H200 1349.3 h -> gpuA100x4-interactive, 0.95 h at 4.0 SU/h (max 3.8); SU used 0.0, reserved 0.0
