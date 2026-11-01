# Post-midterm 1,024-token EgoNormia diagnostic

All four frozen Qwen3.5-9B conditions completed the exact user-supplied 400-ID
correct-video panel (348 source videos) at a 1,024-token response cap. Slurm
array `50085446` and dependent analysis `50085448` completed successfully.
The only intended change from the 384-token runs was the generation cap.
The [protocol](CAP1024_PROTOCOL.md) records the design. All failures remain
in the denominator; these are repeated evaluations on a previously used
development panel, not a fresh blind test.

| Model | Joint 192 | Joint 384 | Joint 1024 | Invalid examples 384→1024 | Justification cap hits 384→1024 |
|---|---:|---:|---:|---:|---:|
| Base | 23.8% | 30.8% | 31.2% | 48→22 | 29→9 |
| RLOO | 26.2% | 33.5% | 33.8% | 42→21 | 29→11 |
| GRPO | 19.2% | 27.8% | 29.0% | 42→17 | 34→9 |
| Answer-only SFT | 28.0% | 35.0% | 35.2% | 41→17 | 34→9 |

At 1,024 tokens, action / justification / joint accuracy and sensibility IoU
were 48.0 / 41.8 / 31.2 / 48.8% for Base; 48.2 / 44.0 / 33.8 / 49.0%
for RLOO; 43.8 / 42.8 / 29.0 / 48.2% for GRPO; and 50.0 / 41.8 /
35.2 / 48.7% for SFT. Paired 384→1024 joint changes were +0.5 points for
Base (95% source-video-cluster interval −0.5 to +1.5), +0.25 for RLOO
(−0.5 to +1.2), +1.25 for GRPO (0.0 to +2.8), and +0.25 for SFT
(−0.8 to +1.3). RLOO exceeded Base by +2.5 points at 1,024 tokens
(−1.0 to +6.0), GRPO trailed Base by −2.25 (−6.4 to +1.7), and SFT
exceeded Base by +4.0 (+0.7 to +7.2). These intervals describe example
sampling, not training-seed variability.

Increasing the cap from 192 to 384 recovered about seven joint-accuracy
points across models; increasing it from 384 to 1,024 recovered only 0.25–1.25.
Invalid outputs and justification cap hits fell markedly at 1,024, but the
ranking stayed SFT > RLOO > Base > GRPO. Thus, truncation remains a measurable
formatting issue, yet more generation alone does not explain the bulk of
remaining EgoNormia errors. Nine to eleven justifications per model still hit
the 1,024-token cap, so the diagnostic does not establish a fully
untruncated ceiling.

The 384-token cap was originally chosen as a simple doubling of the frozen
192-token pilot setting after a 40-item ablation revealed truncation. It was
never selected as an optimal reasoning length. The present 1,024-token
experiment directly tests that concern. The substantive next question is
whether a reward that includes visually grounded **normative** judgments can
transfer better than R³-FDT final-answer correctness alone.

Machine-readable summaries are under `reports/results/cap1024_*`.
Raw predictions remain on Ocean at
`/ocean/projects/cis260338p/apatawar/social_norm_rl/runs/*_ego_cap1024_video400`.
