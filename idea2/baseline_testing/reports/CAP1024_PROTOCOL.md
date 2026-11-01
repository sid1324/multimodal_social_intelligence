# Post-midterm 1,024-token response-cap diagnostic

The original 192-token response cap was the fixed midterm comparison. A
384-token cap was chosen as a simple doubling after observing truncated
justification answers on the first 40 EgoNormia items; it was not an optimized
or theoretically motivated length. On the full 400-ID panel, 29–34 of 400
justifications still reached 384 tokens in each model condition. The 1,024-token
cap is a further diagnostic, not a newly selected model or a blind benchmark.

Evaluate the same frozen Qwen3.5-9B base, RLOO-100, GRPO-100, and SFT-100
checkpoints on the exact user-supplied 400-ID correct-video EgoNormia panel.
Keep the same eight-frame preprocessing, official prompts, generation mode,
seed, native parsing, and attempted-item denominator. Change only
`--max-new-tokens` to 1024. Each of the four jobs requests one V100-32 GB and
writes resumable predictions to its own output directory. The job array is
limited to one simultaneous task to use one V100 at a time.

After all four runs cover 400/400, compute EgoNormia action, justification,
joint accuracy, sensibility IoU, invalid-output rates, and cap-hit rates.
Compare within-model 384→1024 changes and model-vs-base differences with
paired source-video-cluster intervals. Keep failed/invalid outputs in the
denominator. Report 192, 384, and 1024 on the same panel, emphasizing that
all cap comparisons are post-midterm exploratory and share a previously used
development panel. The experiment tests truncation sensitivity; longer
generation does not guarantee more faithful visual or social reasoning.
