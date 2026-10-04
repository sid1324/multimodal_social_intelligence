# Idea 3 / Analysis 2: Testing the data premises of physiognomic-bias counterfactuals

Milestone 2 analysis by Yash Lothe (Team 23). Report section: "Analysis 2: Testing the Data Premises of Physiognomic-Bias Counterfactuals" and Appendix B.

## What is here

| Folder | Contents |
|---|---|
| `code/` | Analysis scripts used for the reported results (see below). |
| `results/` | Human labels, label samples, pilot and trait-check model outputs, summaries. |
| `figures/` | Figure 2 of the report (PDF and PNG). |
| `earlier_pipeline/` | Earlier screening, consolidation and face-editing pilot code (paths may need adapting). |

## Scripts (`code/`)

| Script | Purpose | Output in `results/` |
|---|---|---|
| `text_arm.py` | Lexicon flagging person-directed answers (P3) | `text_arm_summary.json`, `text_arm_items.csv` |
| `blur_score.py` | Sharpness score for each face detection; SHA-256 check of source images | `face_scores.csv`, `face_scores_meta.json` |
| `label_server.py`, `label_round2.py` | Local, score-blind labelling pages used for the human labels | `label_sample*.json`, `human_labels*.json` |
| `baseline_released.py` | Released-answer baseline over 8 VLMs (source-clustered bootstrap) | `released_baseline.json` |
| `pilot.py` | Text-cue pilot (Gemini 3.1 Pro, 60 items x 5 conditions) | `pilot_*.jsonl`, `pilot_*_summary.json`, `pilot_disagreement.json` |
| `legibility.py` | Trait-legibility ratings of the edited scenes | `legibility_*.jsonl`, `legibility_*_summary.json` |
| `figure_final.py` | Figure 2 | `figures/` |
| `analyze.py` | Superseded calibration attempt for the sharpness score (reported as a negative result) | - |

The scripts expect to be run from a working folder containing `data/` and `results/` subfolders.

## Data not included

To respect dataset terms and the privacy of people in the videos, this folder contains **no EgoNormia/Ego4D images, face crops or edited faces**, and no API keys. Re-create the inputs from the pinned sources:

- EgoNormia annotations, commit `09d8a7c53f06ed582236722ba8496c6a504cd1ab`: `src/final_dataset/final_data.json` (saved as `data/official_annotations.json`) and released model answers `src/final_dataset/final_data_eval.json` (saved as `data/released_predictions.json`), from https://github.com/open-social-world/EgoNormia
- Preview frames, dataset revision `2937df7fa96d8515417e7b5122fbf8eaeeaeee86`: run `earlier_pipeline/screening/download_all.py`, then `screen_all.py` for `data/screening_results.json`.
- Model access for `pilot.py` and `legibility.py`: a `.env` file with `LITELLM_BASE_URL` and `LITELLM_API_KEY` (never commit it).

## AI use

See Appendix D of the report.
