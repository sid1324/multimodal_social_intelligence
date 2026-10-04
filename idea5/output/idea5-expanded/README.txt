EgoNormia Idea 5: expanded frozen-model experiment

Open review.html directly in a browser once reporting completes. If a chat link does not open, use Finder > Go > Go to Folder and paste:
/Users/patrickjain/Desktop/CMU/sem3/multimodal/output/idea5-expanded

Models:
Qwen3-VL-32B-Instruct community 8-bit MLX checkpoint; local M5 Max / 128 GiB.
MediaPipe Pose Landmarker Full and Hand Landmarker.
YOLOv8n-seg (COCO instance segmentation).
Depth Anything V2 Small (relative monocular inverse depth; not measured distance).
GPT-5.6-sol requested via spawned agent: five-example blinded workflow pilot, not stateless API model comparison.

Sample: same fixed 40 distinct source videos and 200 pre-action frames as ../idea5-audit. No outcome filtering. All conditions retain missing detections.
Conditions: RGB; RGB+pose; RGB+segmentation; RGB+depth; RGB+all three; repeated RGB (one and three extra slots); shuffled-all. Each is run with images alone and with descriptions, totaling 640 paired result rows / up to 1280 action/justification calls.
RGB captions see only the original grid, without question/options, labels, norm taxonomy, or dataset descriptions. Processed captions deterministically summarize extractor outputs. Caption lengths are not controlled.

Key files after completion:
review.html: all image grids with attached descriptions, results, controls, and review-note export.
analysis-preview.pdf: standalone draft, not the complete team report.
idea5-expanded-section.tex: insertable section for the existing Overleaf report.
summary.json: full metrics and paired intervals.
qwen_results.json: raw prompts, choices, timing and mapped/scored results; resumable per condition.
gpt56-agent-results.json: blinded pilot predictions and image-view evidence.
gpt56-expanded-blind-manifest.json: exact unchanged inputs supplied to pilot agent.
gpt56-agent-scored.json: scoring performed externally after agent completion.
inputs.json: image paths and every attached description.
geometry_results.json / geometry_protocol.json: processed outputs, model revision/hash and settings.
model.json / qwen_protocol.json: model and inference protocol.
geometry-verification.json / verification.json: checks actually performed.

Reproduce from workspace root:
.venv-geometry/bin/python scripts/extract_egonormia_geometry.py
.venv/bin/python scripts/prepare_expanded_inputs.py
.venv/bin/python scripts/compare_expanded_qwen.py
.venv/bin/python scripts/build_expanded_report.py
(cd output/idea5-expanded && tectonic analysis-preview.tex)

Student review and interpretation remain pending. Preserve all AI disclosures, confirm assignment scope with the TA, and integrate with teammates' analyses and the shared page limit. The original 8B run is preserved separately; its prompt differs and does not establish a size-only comparison.


LATEST STATUS - USER STOPPED RUN
All inference and watchers were terminated at user request on October 4, 2026. Do not resume automatically. Saved 562 rows; matched report uses the first 35 fully evaluated clips (560 rows). Two additional rows remain preserved. Report, gallery, chart, CSV and TEX are compiled; verification passed and all four PDF pages were visually inspected. See RESULTS.md. Prior instructions about live PIDs or waiting for 640 rows are superseded.
