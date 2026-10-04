# Idea 5: explicit geometry for social-norm reasoning (Analysis 3)

Everything behind Analysis 3 of the second project assignment: the code, the raw model answers, the scored
results, the report sources, and the exploratory work that came before the final experiment.

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
