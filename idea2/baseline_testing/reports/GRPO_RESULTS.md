# Post-midterm GRPO follow-up: measured outcome

The GRPO follow-up completed on one V100-32 GB. This is an exploratory comparison requested after the midterm RLOO, SFT, and EgoNormia results were known. It reuses the same 400-ID EgoNormia panel for evaluation, so it is **not a new blind test**. No EgoNormia examples or labels entered training. The [protocol](GRPO_PROTOCOL.md) fixes the implementation and comparison scope.

## Training and coverage

The Qwen3.5-9B NF4/FP16 rank-8 language LoRA policy was trained on accessible R³-FDT movie clips, with four sampled completions per question, standardized group rewards, two clipped policy epochs per mixed-reward group, and sampled reference KL. The final adapter is `/ocean/projects/cis260338p/apatawar/social_norm_rl/runs/grpo/checkpoint-00100`. Slurm job `49994657` completed in 2:01:30. It visited 203 training questions across 78 videos: 100 mixed-reward groups trained, 97 uniform-reward groups skipped, and 6 groups skipped after nonfinite gradients in both epochs. There were 199 accepted optimizer updates. Mean binary reward across all 812 sampled completions was 0.700. Mean second-epoch sampled KL was 0.056; mean clipped-token fraction was 0.00032. Peak reserved GPU memory was 12.93 GB. This is **not compute-matched** with the earlier 100-update RLOO run.

## Evaluation

The same model checkpoint was evaluated on 60 movie-disjoint R³-FDT questions and the exact user-supplied 400-ID EgoNormia correct-video panel (348 source videos). Each EgoNormia condition covers all 400 examples, and failed/invalid outputs remain in the denominator. The 192-token and 384-token EgoNormia jobs and their CPU analyses all completed successfully (`49994658`, `49994659`, `49999970`, `49999971`). Scores below are percentages.

| Model | R³-FDT video accuracy, n=60 | EgoNormia joint, 192 tokens | EgoNormia joint, 384 tokens |
|---|---:|---:|---:|
| Starting model | 68.3 | 23.8 | 30.8 |
| RLOO | 80.0 | 26.2 | 33.5 |
| Answer-only SFT | 93.3 | 28.0 | 35.0 |
| GRPO | 83.3 | 19.2 | 27.8 |

At 192 tokens, GRPO's action accuracy was 39.8%, justification accuracy 33.0%, sensibility IoU 45.1%, and joint accuracy 19.2%. The starting model scored 44.8%, 36.2%, 46.5%, and 23.8%, respectively. The paired GRPO-minus-base joint difference was −4.5 percentage points (95% source-video-cluster bootstrap interval −9.2 to −0.2). GRPO produced 184 invalid examples under the evaluator's native parsing, versus 167 for the base model. Its justification task hit the generation cap on 145/400 examples.

At 384 tokens, GRPO's action accuracy was 43.5%, justification accuracy 41.8%, sensibility IoU 48.0%, and joint accuracy 27.8%. The starting model scored 48.2%, 40.8%, 48.4%, and 30.8%. The paired joint difference was −3.0 points (95% source-video-cluster interval −7.5 to +1.3). GRPO had 42 invalid examples, versus 48 for the base model, and justification cap hits fell to 34/400. Extending generation therefore reduced formatting/truncation failures, but did not reverse the observed joint-accuracy deficit.

The R³-FDT video gain of +15.0 points over base has a 95% movie-cluster interval of −2.3 to +36.1 on this small 60-question holdout. GRPO blind accuracy was 68.3%, versus 61.7% for the base model. These figures show that the model can learn the source task; they do not establish stronger use of video, because the no-video score also rose and the holdout is small.

## Interpretation and limitations

This run does not support the hypothesis that this particular R³-FDT GRPO recipe improves EgoNormia transfer. The source-task reward checks final-answer correctness only, while EgoNormia requires a normatively appropriate action, a matching justification, and socially acceptable alternatives. The observed gap is consistent with objective mismatch, but the measurements do not identify a unique causal mechanism. At 192 tokens, many GRPO justification failures also reflect generation-cap and output-format sensitivity; the 384-token repeat limits that explanation because the joint deficit persists while invalid outputs fall.

The RLOO and GRPO runs match the model, accessible source dataset, and EgoNormia panel, but differ in optimization and compute. They each have one training seed, and the 400-ID panel was already used in development. Cluster intervals quantify example-sampling uncertainty only. A fresh, untouched evaluation panel and seed/compute replication are needed for a confirmatory algorithm claim. This follow-up should be labeled **post-midterm exploratory** in any course report or slides.

Machine-readable summaries and the native evaluator outputs are in `reports/results/grpo_*`; raw predictions and the adapter remain on Ocean under `/ocean/projects/cis260338p/apatawar/social_norm_rl/runs/`.
