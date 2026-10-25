# Status — 2026-10-24

## Latest audited midterm outcome

The 400-ID correct-video extension and CPU merge completed successfully:
base 49766293 (260 continuation predictions), RLOO 49766294 (260), SFT
49766295 (260), merge/audit 49766301. Each model has 400/400 correct-video
predictions from 348 EgoNormia source-video groups, with zero inference
exceptions. The merge checks exact supplied ID order, model/adapter settings,
video inputs, and prediction completeness; hashes are in
`reports/results/video400_provenance.json`.

Native-style joint action/justification accuracy is 23.8% base, 26.2% RLOO,
and 28.0% SFT. RLOO minus base is +2.5 percentage points, source-video-cluster
bootstrap 95% interval −0.8 to +5.7; SFT minus base is +4.3 points, +0.5 to
+8.0. SFT minus RLOO is +1.8 points, −2.5 to +5.9. The RLOO Ego transfer
hypothesis remains unresolved and no RL-specific advantage is demonstrated.
The original 140-ID no-video/mismatched-video controls are retained separately.
See `reports/MIDTERM_SECTION.md` for a paste-ready report section and
`reports/MIDTERM_DRAFT.md` for the full research record.

A separately labeled full-panel 384-token diagnostic also completed: base
49771983, RLOO 49771986, SFT 49771988, and merge/audit 49771990. All three
have 400/400 correct-video predictions. Joint accuracy is 30.8% base, 33.5%
RLOO, 35.0% SFT. The within-model 192→384 cap gain is about seven points for
each model; invalid examples fall to 48, 42, and 41 respectively. RLOO minus
base remains uncertain at +2.8 points (cluster interval −0.5 to +6.1), and
SFT minus base is +4.3 (+0.8 to +7.7). Aggregate results and merge hashes
are in `reports/results/cap384_video400_*`.

## Execution history

Implemented: annotation audit and fixed partitions; media acquisition; NF4
language-only LoRA loader; synthetic GPU smoke/reload test; native-prompt
evaluation; RLOO pilot and SFT control; coverage-aware metrics, category
summaries, source-cluster bootstrap, and official scorer export.

CPU checks: 8 protocol tests passed, including scorer parity on synthetic valid
predictions, predicted-action conditioning, label remapping, and split integrity.

Completed: the team supplied an exact 400-ID EgoNormia sample, now frozen as
the midterm panel with source-video group separation from final evaluation.
All 400 IDs have official repository annotations; 376 appear in the HF parquet
and 24 use repository annotations. The original 200-clip pool was downloaded,
but it is superseded by the team sample. Storage and Slurm account migrated to
cis260338p; eight CPU protocol tests passed after the change.

Completed: pinned official model download on CPU compute job 49742706. All
four shards passed SHA-256 verification and the model directory has a READY
marker. All 400 team-supplied preceding
clips were validated and the exact-order accessible manifest is complete.
The isolated environment has passed import and SM70 checks and has a
READY marker. The initial R3 movie pool finished with
96/100 validated videos, giving 1,262 train, 86 val, and 79 test questions.
The installed torch 2.8.0+cu126 binary reports compiled `sm_70` support.
A CPU-only preflight with the pinned Transformers 5.3.0 processor successfully
formatted synthetic video for EgoNormia action, justification, and sensibility
prompts (record: storage `runs/processor_preflight.json`).
The pinned Qwen3.5 model class, BitsAndBytes configuration, 4-bit layer, and
PEFT LoRA imports also succeeded on CPU.
Slurm readiness watcher 49730224 completed. The first one-V100 smoke job
49730226 loaded the model in about 231 seconds, then failed in video position
assignment: Transformers 5.3.0 generated four video-token runs from one
eight-frame video but supplied one video-grid row. The input path now presents
the same ordered frames as four two-frame chunks, giving four token runs and
four grid rows; CPU processor preflight passed. Replacement smoke 49743943
passed on a single V100-32 GB: real R3 video generation, finite adapter
gradients, optimizer step, and adapter reload parity. Peak reserved memory was
13.01 GB and total job work after start took about 243 seconds. Baseline
49743945 and CPU analysis 49743947 completed successfully. RLOO train
49743946 and adapted evaluation 49743948 subsequently completed. An
exploratory 384-token cap ablation on the first 40 frozen panel IDs ran as job
49754768, separately from the primary 192-token comparison. An answer-only
SFT control completed in job 49754875 using the same R3 training source and
frozen midterm evaluation panel.
See ../HANDOFF.md for processes, logs and execution order.

The base-model EgoNormia evaluation completed on 140/400 team-supplied IDs
from 133 source videos, with all three conditions paired. Native-style joint
accuracy: 28.6% correct video, 10.0% no video, 7.1% mismatched video.
Action, justification, and sensibility metrics, uncertainty intervals,
category tables, invalid-output counts, and case sheets are under
`reports/generated/baseline/` and `reports/generated/error_review/`.
Held-out R3 baseline accuracy: 62.5% on 40/86 validation questions, 68.3%
on 60/79 test questions with video, and 61.7% on the same 60 test questions
without video. All are measured on the accessible movie-disjoint pool.

RLOO job 49743946 completed its requested 100 optimizer updates in 84.1
minutes and saved `runs/rloo/checkpoint-00100`. It sampled 156 R3 training
questions from 73 videos/70 movie groups. The trace has 49 uniform-reward
skips, seven early nonfinite-gradient skips, 624 completions, and 26
completion-cap hits. Mean sampled correctness was 58.7%; mean sampled KL on
update candidates was 0.0108. A coverage summary and plot are in
`reports/generated/rloo_training/`.
The first adapted evaluation job 49743948 completed 116 matched EgoNormia
triples plus one additional video item before its 2.2-hour inference cap.
Resumable continuation 49760681 completed the exact 140-ID panel. The cap
ablation depended on the continuation, keeping project V100 jobs sequential.
Comparison tables use paired IDs and report coverage.
Held-out R3 adapted accuracy: 70.0% on 40 validation questions; 80.0%
correct-video and 61.7% no-video on the same 60 test questions. This is
+11.7 points versus base on video test, with only five source groups.

The adapted continuation completed all 140 matched EgoNormia triples.
Correct-video joint accuracy is 31.4% versus the base model's 28.6%, a paired
+2.9-point difference with source-group bootstrap 95% interval −2.2 to +8.5.
Action accuracy is unchanged at 47.1%; justification rises from 40.7% to
44.3%; sensibility IoU remains about 46.3%. This is an uncertain positive
point estimate, not demonstrated improvement. The exploratory cap ablation
job 49754768 completed on the first 40 fixed IDs: base-model joint accuracy
rose from 7/40 at 192 tokens to 12/40 at 384 tokens, while invalid examples
fell from 19 to 4. This is a separate diagnostic, not the primary protocol.
The answer-only SFT control job 49754875 completed 100 updates from 113 R3
training questions in about 5.3 minutes, then scored the same 140 EgoNormia
items. Its correct-video joint score is 32.1% versus 31.4% RLOO and 28.6%
base; paired uncertainty includes zero for both SFT comparisons. On the 60
held-out R3 test questions it reaches 93.3%. This does not establish an
RL-specific transfer advantage. Matching RL-adapter 384-token job 49764791
completed the small model-by-cap comparison on the first 40 EgoNormia IDs:
at 192 tokens, base 7/40 and RLOO 10/40 joint correct; at 384 tokens, base
12/40 and RLOO 13/40. Invalid examples fall sharply for both models.
See `reports/results/cap_factorial_40.json` for paired intervals and exact
counts. This exploratory slice is separate from the 140-ID primary result.

Additional correct-video coverage completed on the remaining 260 supplied
EgoNormia IDs: base job 49766293, RLOO job 49766294, and SFT job 49766295,
sequentially on one V100. CPU merge/audit job 49766301 verified identical
settings and assembled the 400-ID video-only comparison above. The original
140-ID three-condition analysis remains intact.
