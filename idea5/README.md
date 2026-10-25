# Idea 5: explicit geometry for social-norm reasoning

Everything behind Idea 5: Analysis 3 of the second project assignment (4 October 2026) and the midterm report
(25 October 2026, see "Midterm report" at the end). That covers the code, the raw model answers, the scored
results, the report sources, and the exploratory work that came before each experiment.

# Analysis 3 (second assignment)

The question: before training anything, does a frozen vision-language model get anything out of explicit
geometry (depth and segmentation maps) beyond the RGB frames those maps are derived from?

## Result

Sixty EgoNormia items, six frozen models, one isolated call per item and input. An answer counts only if the
action and its justification are both correct. Numbers are correct answers out of 60.

| Model | Options only | + depth (no RGB) | + frames | + frames, three times | + frames, depth, segmentation |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-32B (8-bit, local) | 17 | 22 | 38 | 38 | 38 |
| GPT-5.5 | 25 | 27 | 41 | 40 | 42 |
| GPT-5.6 | 21 | 29 | 37 | 39 | 41 |
| GPT-6 | 23 | 28 | 42 | 46 | 42 |
| Claude Sonnet 5.5 | 33 | 31 | 46 | 48 | 45 |
| Claude Opus 5.5 | 32 | 42 | 45 | 45 | 45 |
| All six (of 360) | 151 | 179 | 249 | 256 | 253 |

- Text alone is not enough: frames add 27 points on average (95% interval 16 to 38; significant for every model).
- A depth map without RGB adds about 8 points over text (interval -0.3 to 16; five of six models improve).
- Depth and segmentation on top of frames add nothing: +1.1 points on average (interval -1.7 to +4.2), 17 answers
  corrected and 13 broken across 360 model-item pairs.
- Repeating the same frames three times does as well as adding the maps, and for Qwen the maps of a different
  item do as well as its own (39 vs 38).
- The 22 Proxemics items score 91 of 132 with or without maps.

This does not test the idea itself. The models were not adapted to these inputs; fine-tuning on them is the
next experiment.

## Layout

The folder mirrors the working directory the experiments ran in, so the scripts run from `idea5/` once the
data and models are in place (see "Not in the repository").

| Path | What it is |
|---|---|
| `scripts/clean60_prepare.py` | Picks the 60 sharpest items of the verified split (one per source video), builds the RGB, depth and segmentation grids, writes the task file and answer key |
| `scripts/clean60_nn.py`, `clean60_qwen_nn.py` | Final run: options only, + frames, + frames with maps, with prompts that rule out the "none of these" option |
| `scripts/clean60_repeat.py`, `clean60_qwen_controls.py` | Controls: the RGB grid three times (all models) and maps from another item (Qwen) |
| `scripts/clean60_depth_only.py` | Options plus depth maps, no RGB |
| `scripts/clean60_score_nn.py`, `clean60_score_repeat.py`, `clean60_score_depth_only.py`, `clean60_by_category.py` | Scoring, paired tests, per-category breakdown |
| `scripts/analysis3_inputs_figure.py` | The figure in the report |
| `scripts/clean60_run.py`, `clean60_gpt_more.py`, `clean60_claude_text.py`, `clean60_claude_images.py`, `clean60_qwen.py`, `clean60_score_all.py` | First run, where "none of these" was allowed (not reported; kept because later scripts import from them) |
| `scripts/language_exploration.py` | Text analysis of all 1,853 items: the spatial word count used in the report, plus BERT and Empath probes that are not reported |
| `scripts/clean60_hard_items_page.py`, `clean60_gpt_tool_audit.py` | Inspection page for the items every model misses; check that the GPT calls used no tools |
| `scripts/download_egonormia.py`, `audit_egonormia_pose.py`, `extract_egonormia_geometry.py`, `compare_qwen_pose.py`, `compare_expanded_qwen.py`, `prepare_*.py`, `build_*.py`, `verify_expanded_experiment.py`, `finish_expanded_report.py` | Earlier exploratory work on a different 40-item sample (pose audit, Qwen3-VL-8B and 32B pilots); not reported |
| `output/idea5-report/` | Report sources: `analysis3_simple.tex` is the submitted section, `implications.tex` the discussion paragraph, `analysis3_inputs.pdf` the figure; other files are earlier drafts |
| `output/idea5-clean60/` | The 60-item experiment: `tasks.json`, `answer_key.json`, `selection.json`, the image grids, every raw answer under `results/`, and the scored summaries (`scored_nn.json`, `repeat_control_scored.json`, `depth_only_scored.json`, `by_category_nn.json`) |
| `output/idea5-language-v2/` | Text analysis outputs (`summary.json`, figures) |
| `output/idea5-language/` | First version of the text analysis, which assumed the empty option is always the fifth one; superseded by v2 |
| `output/idea5-claude/` | An early 15-item pilot with Claude agents answering in batches; superseded by the isolated-call runs |
| `output/idea5-audit/`, `output/idea5-expanded/`, `output/idea5/` | The earlier exploratory work: pose availability, Qwen pilots, review pages, a handoff note (`idea5-expanded/HANDOFF.md`) |

## How the models were called

- Claude Sonnet 5.5 and Opus 5.5: the Claude Code CLI in print mode with all tools disabled, images sent inline,
  one process per answer.
- GPT-5.5, GPT-5.6, GPT-6: the Codex CLI (`codex exec`) in an empty read-only sandbox, one process per answer;
  the event log of every scored call in the final run shows a single model message and no tool call.
- Qwen3-VL-32B: the 8-bit MLX build, run locally with greedy decoding.

All runs are from 4 October 2026.

## Reproducing

```
python scripts/clean60_prepare.py                 # needs data/EgoNormia and the depth and segmentation models
python scripts/clean60_nn.py claude               # Opus and Sonnet
python scripts/clean60_nn.py gpt gpt-5.5 gpt-5.6-sol gpt-6-sol
python scripts/clean60_qwen_nn.py                 # needs models/Qwen3-VL-32B-Instruct-8bit and mlx-vlm
python scripts/clean60_repeat.py claude ; python scripts/clean60_repeat.py gpt gpt-5.5 gpt-5.6-sol gpt-6-sol
python scripts/clean60_qwen_controls.py
python scripts/clean60_depth_only.py claude       # then: gpt ..., qwen
python scripts/clean60_score_nn.py ; python scripts/clean60_score_repeat.py ; python scripts/clean60_score_depth_only.py
python scripts/analysis3_inputs_figure.py
```

Model answers are not deterministic for the API models, so a rerun will not match item for item.

## Not in the repository

- The EgoNormia data (about 65 GB): https://huggingface.co/datasets/open-social-world/EgoNormia, released under
  CC BY-SA 4.0. The image grids here are derived from it.
- Model weights (Qwen3-VL, Depth Anything V2, YOLOv8n-seg, MediaPipe).
- Per-frame images and raw depth arrays from the exploratory runs (about 530 MB) and the BERT embedding caches.
  The scripts regenerate them. The review pages under `output/idea5-audit/` and `output/idea5-expanded/` link to
  some of these files and will show gaps.
- Some JSON files contain absolute paths from the machine the experiments ran on.

## Caveats

- Sixty items detect only large effects; each accuracy is uncertain by about 12 points either way.
- The items were chosen for sharp frames, which is a best case for the depth and segmentation models.
- The statistics were checked by the assistant that produced them and reviewed by the author; an independent
  verification was started and stopped before it finished.
- AI use is described in the report's disclosure section. The code, runs and drafts here were produced with
  OpenAI Codex and Claude Code under the direction of Shiv Gupta.

# Midterm report

Everything behind the Idea 5 section of the midterm report (25 October 2026), including work that was run but
not reported. Code is in `scripts/` (`midterm_*`, `lora_pilot/`, `emb_ft/`, `rl_pilot/`, `lit_scaling_analysis.py`,
`frontier_spatial_social.py`); data and results are in `output/idea5-midterm/`.

## Question and approach

Analysis 3 found that frozen models get nothing from depth and segmentation maps beyond the RGB frames. The
midterm asks whether *adaptation* changes that: does giving a model explicit spatial structure, or training it on
spatial tasks, improve social-norm reasoning on EgoNormia beyond matched non-spatial controls?

The core idea, taken from the reading-week literature on universal multimodal embeddings, is to put everything
about an item into one embedding: the RGB frames, a text description of them, depth, segmentation and pose masks,
and text descriptions of each mask written by another model. Every spatial condition is paired with a control that
carries the same amount of non-spatial input, and the difference is called a spatial-specific effect.

## Data and protocol

- Test set: the 200 EgoNormia verified items (`splits.json`), where "None of these" is never the correct answer.
  600 items from disjoint videos are used for training and cross-validation (800 in total).
- Metric: joint accuracy, both the action and its justification correct.
- Statistics: paired by item, exact McNemar tests, 95% bootstrap intervals, and DerSimonian-Laird random-effects
  pooling of the ten primary spatial-specific contrasts (`midterm_stats.py`, `midterm_meta.py --no-rl`).
- Test labels were never sent to the remote GPU machines or used for training; all test scoring ran locally.

## What was done

### 1. Spatial inputs (`midterm_prepare_maps.py`, `midterm_prepare_pose.py`, `midterm_describe.py`, `midterm_flip_desc.py`)

- For each item's five-frame strip: depth (Depth-Anything-V2-Small), instance segmentation (YOLOv8n-seg) and
  body/hand skeletons (MediaPipe), saved as image grids and raw landmarks (`pose_landmarks/`).
- Gemini 4 wrote a question-blind description of the RGB strip and of each map (`descriptions/`).
- Two text variants for a counterfactual test: a reworded copy keeping every fact (`descriptions_paraphrased/`)
  and a copy with spatial relations reversed (`descriptions_flipped/`).
- Four spatial labels from the extractors for probing: people count, nearest-person distance, facing, hands
  visible (`spatial_labels*.json`).

### 2. Universal embedding with a trained scorer (`midterm_embed*.py/.sh`, `midterm_scorer.py`)

- Qwen3-VL-Embedding-8B (frozen) embeds the context and each answer option under several conditions: C2 = RGB +
  description; C5 = C2 + all masks and their descriptions; X1 = C5 with masks and descriptions from another item;
  X3 = C2 with the RGB repeated (plus C0, C1, C3, C4, C6, X2 variants in `emb/`).
- A low-rank bilinear scorer with an option prior and "None" bias is trained on the vectors with 5-fold
  video-grouped cross-validation over all 800 items; a logistic probe predicts the four spatial labels.
- Result: the masks are encoded (probe accuracy +3.4 pp [+2.2, +4.7], C5 vs C2) and C5 beats C2 on social accuracy
  (+2.6 [+0.4, +4.9]), but not the mismatched-mask control (C5 - X1 +1.3 [-0.9, +3.4]); C5 - X3 is +2.1
  [+0.4, +4.0].

### 3. Zero-shot answering by frontier and open models (`midterm_direct_eval.py`, `midterm_direct_score.py`)

- Gemini 4, GPT-5.6 and Qwen3-VL-32B, one isolated call per item, on: RGB only, RGB + maps, mismatched maps, RGB
  repeated, RGB + descriptions, mismatched descriptions, text descriptions only (original, reworded, spatially
  reversed), EgoNormia's own scene description, spatial vs generic step-by-step prompts, and forced choice with
  "None" removed (`direct_eval/`).
- Results (3 models pooled): maps beat RGB alone (+3.4 [+0.9, +5.9]) and RGB repeated (+2.8 [+0.3, +5.2]), but
  correct maps do no better than mismatched ones (-0.7 [-3.3, +1.8]). Text-only with true spatial facts beats
  reversed facts (+3.5 [+0.9, +6.0]), the only spatial-specific gain found. Spatial reasoning prompts are no better
  than generic ones (GPT -2.5, Qwen -4.0, both n.s.).

### 4. Abstention and error analysis (`midterm_error_types.py`, `midterm_spatial_need.py`)

- Most differences between input conditions are shifts in how often a model picks "None". EgoNormia's own
  (non-spatial) scene description costs GPT-5.6 9 pp, all through extra "None" picks; a mismatched description
  costs Qwen3-VL-32B 22 pp. With "None" removed, conditions converge (GPT-5.6 71.5-73.0%), and Gemini 4's gain from
  maps comes from picking "None" less often.
- A GPT-5.6 judge, checked by Gemini 4, classified every wrong answer: about one error in 63-87 per model involves
  spatial geometry; about 60% are perception failures and 20-31% misjudged norms (`error_types/`).
- GPT-5.6 rated how much spatial understanding each item needs; items rated high-need gain no more from maps than
  low-need ones (-1.4 [-4.6, +1.9]) (`spatial_need.json`).

### 5. Spatial LoRA fine-tuning on Bridges-2 (`lora_pilot/`)

- Qwen3-VL-8B with LoRA (r = 16) on 5,000 SAT spatial QA items, against a LoRA on 5,000 non-spatial VQAv2 items with
  the same format and budget, two seeds each; a third, narrower control trained only on VQAv2 colour questions.
- The spatial LoRA learns its task (held-out SAT 78.8% -> 88.8%) but scores 6.5 pp below the non-spatial control on
  EgoNormia (-6.5 [-12.0, -1.0], p = .003, two seeds pooled), and answers its action and justification
  consistently less often (82.5% vs 91.0%). The colour-only control collapses (35.5% joint).

### 6. Spatial LoRA x combined input, Runpod RTX 5090 (`lora_pilot/eval_combined.py`, `run_combined_pod.sh`)

- 2x2: base or spatial-LoRA Qwen3-VL-8B, given RGB only or the combined input (maps + descriptions).
- Joint accuracy: base RGB 60.5%, base combined 56.0%, spatial LoRA RGB 55.5%, spatial LoRA combined 50.8%. The
  spatial LoRA with the combined input is 9.8 pp below base on RGB ([-16.5, -3.2], p < .001); the adapter and the
  extra input each cost about 5 pp and simply add (interaction -0.2).

### 7. Fine-tuning the universal embedding itself, Runpod A100 (`emb_ft/`, not in the report)

- End-to-end LoRA on Qwen3-VL-Embedding-8B (r = 16, vision frozen, contrastive loss over the option embeddings,
  tau = 0.05) on the 600 training items, for C2, C5 and X1, plus a zero-shot pass (`output/idea5-midterm/emb_ft/README.md`).
- Fine-tuning lifts every condition by 11-19 pp over zero-shot. C5 beats C2 (+6.0 [+2.0, +10.6], p = .012, n = 199)
  but exactly ties the mismatched-mask control X1 (+0.0 [-3.5, +3.5]); Proxemics items show no gain either.
- Seed 0 only: the seed-1 runs were stopped when the pod was closed.

### 8. Published spatial fine-tunes (`midterm_direct_eval.py`, `paper_models.json`)

- Each against its own base model on the same 200 items: SpatialLadder-3B -12.0 [-19.5, -4.5] (p = .003),
  MindCube-RL-3B -5.5, SpaceR-7B -0.5, Spatial-SSRL-7B +1.5. The models with the largest reported spatial gains
  lose the most on social norms.

### 9. Reinforcement-learning pilot on Bridges-2 (`rl_pilot/`, dropped from the report)

- GRPO with LoRA on Qwen3-VL-8B, trained on EgoNormia training items. Arms: R0 rewards a correct answer only; R1
  asks for a block of spatial facts before reasoning and rewards answer + 0.5 x spatial facts; R2 uses the same
  spatial format but rewards the answer only; R3 rewards the spatial facts only.
- R0 beats the untrained model by +8.3 pp [+4.0, +12.8] after 90 steps, but that comes from training on
  benchmark-style questions. At a matched 40 steps the spatial arms are worse than R0 (R1 -7.0 [-13.5, -0.5],
  R2 -6.5 [-12.0, -0.5]), and spatial-fact accuracy does not improve (53% vs 53%).
- Dropped because training on the benchmark's own kind of data cannot isolate spatial-to-social transfer. R3 and
  the second R1 seed were still running when the work stopped and were never scored.

### 10. Cross-model evidence from published leaderboards (`lit_scaling_analysis.py`, `frontier_spatial_social.py`)

- 30-model composite: spatial and social scores correlate (Spearman 0.69), but only 0.15 after controlling for
  MMMU. Across 284 OpenCompass models, MMBench Social Relation correlates with Spatial Relationship (0.62 raw,
  0.25 partial) about as much as with OCR (0.63, 0.31) or celebrity recognition (0.59, 0.36).
- The report figure: ten models on both the OpenCompass spatial leaderboard (BLINK, 3DSRBench, MMSI-Bench,
  LEGO-Puzzles) and EgoNormia Table 5. Every value was re-downloaded and checked against archived sources
  (`lit_scaling/sources/`, `frontier_spatial_social.csv`). Shown without a fitted line, as supplementary context.

### 11. A bug caught along the way

- Running the embedding model on Apple MPS in half precision with SDPA attention produced wrong embeddings. It was
  caught by checking against the model card's reference outputs, and every affected run was redone in bf16 with
  eager attention; the bad runs are kept out of the repository (`*_INVALID`).

## Result

No spatial intervention beat its matched control: pooling the ten spatial-specific contrasts gives +0.2 pp
(95% CI -1.7 to +2.1), and spatial training lowers social accuracy. The updated research idea is to test whether
spatial and social ability are related at all, across model scales (1B-72B), families, adaptation methods, spatial
data types and social benchmarks, using equivalence tests.

## Report files

`output/idea5-midterm/report/`: `idea5_section_final.tex` (the section), `idea5_section_cited.tex` (with
citations), `forest_compact.pdf` and `frontier_spatial_social.pdf` (figures), `abstract.tex`,
`idea5_ai_usage.tex`, `idea5_contribution.tex`, `idea5_refs.bib`. Other `.tex` and figure files are earlier drafts.
`PLAN.md` is the running experiment plan.

## How the models were run

- Gemini 4 through the Antigravity CLI (`agy`) and GPT-5.6 through the Codex CLI (`codex exec`), one isolated call
  per item.
- Qwen3-VL-32B and Qwen3-VL-Embedding-8B locally on Apple silicon; Qwen3-VL-8B LoRA training and evaluation and the
  RL pilot on PSC Bridges-2; the 2x2 evaluation on a Runpod RTX 5090; the embedding fine-tune on a Runpod A100.
- The code, runs and drafts were produced with Claude Code (Claude Opus 5.5), with Gemini 4 helping to edit the
  prose, under the direction of Shiv Gupta.

## Not in the repository (midterm)

- Embedding arrays (`*.npy`), LoRA adapter weights (`*.safetensors`, 167 MB each) and all image grids (`grids/`,
  `lora_pilot/egonormia_images/`, about 620 MB). The scripts regenerate them.
- The test answer key (`lora_pilot/answer_key.json`); `scripts/lora_pilot/make_egonormia_tasks.py` rebuilds it from
  the EgoNormia data.
- The runs invalidated by the MPS bug (`*_INVALID`).
- Logs contain absolute paths from Bridges-2 and the local machine.
