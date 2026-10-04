# Idea 1 / Analysis 1: Progressive perceptual grounding

Milestone 2 contribution for Team 23. This folder contains the rechecked analysis, code, results, editable report sections, and the completed 20-item human exercise.

## Start here

- [Main report preview](latex/analysis1_preview.pdf) and [editable section](sections/analysis1_section.tex).
- [Human exercise](latex/human_exercise.pdf): one participant, 20 items, 60 responses.
- [Strict review](qa/STRICT_GRADER_REVIEW.pdf) and [review text](qa/STRICT_GRADER_REVIEW.md).
- [Introduction, discussion, and AI disclosure](latex/intro_discussion_disclosure.pdf).
- [Team integration snippets](team_insert/) and [supplement source](latex/supplement.tex).

## Results and limits

Matched description-to-grid joint accuracy gains are 27.1 / 16.2 / 19.0 percentage points for Gemini Flash / Gemini Pro / GPT-4o. These rescore released predictions; an evaluation-code prompt caveat prevents a clean causal information-loss interpretation. No curriculum training or fresh proprietary VLM inference was performed.

The human exercise yields **8/20, 8/20, and 9/20** agreement with released gold for options, descriptions, and descriptions plus frames. Frames corrected three answers and lost two prior agreements. Two ten-item blocks were collected in fixed stage order, with feedback on the first block before the second. Evidence categories were assigned afterward by one AI coder. This is exploratory; there is no inter-person agreement estimate.

The team reports contacting the VideoNorms authors and expects access soon. Annotations have not been received and annotation analysis remains pending.

## Layout

| Path | Contents |
|---|---|
| `scripts/`, `run_all.sh` | Analysis, source restoration, scoring, and validation |
| `outputs/tables/` | Computed results, item IDs, predictions, transitions, evidence counts |
| `outputs/figures/` | Plots without source video imagery |
| `sections/`, `latex/`, `team_insert/` | Report sources, eligible compiled PDFs, and integration files |
| `human_simulation/` | Submitted responses, scores, evidence codebook and annotations |
| `data/folds/`, `data/media_sample_ids.json` | Frozen splits and selection IDs |
| `outputs/logs/` | Public media download manifest and model revisions only |
| `qa/` | Audits, provenance, validation evidence, repository manifest |
| `legacy/` | Historical scripts; earlier probe discrepancies are documented in the review |

## Source data and reproduction

Consistent with the repository's policy, datasets, video frames, contact sheets, model weights, feature caches, private assignment/context files, and the full private ZIP are not committed. Two generated tables reproducing source text (`normlens_items.csv`, `hf_github_field_differences.csv`) are also omitted. The illustrated supplement and qualitative image figure are generated locally. The original full artifact package's archive checks in `qa/` describe that package; see `repository_validation.json` for this distribution's checks.

Use Python 3.13 and the tested versions in `requirements.txt`:

```sh
cd idea1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Restore only the pinned core annotations/predictions, then verify human results:
.venv/bin/python scripts/download_sources.py
.venv/bin/python scripts/11_human_results.py
.venv/bin/python scripts/12_human_validate.py

# Full analysis: downloads external sources, 400 strips, and pretrained weights.
# This requires network access, substantial disk space, and CPU/GPU compute.
.venv/bin/python scripts/download_sources.py --external --media --models
ANALYSIS_PYTHON="$PWD/.venv/bin/python" bash run_all.sh
```

The default download verifies core hashes. `--external` restores the HF parquet, NormBank CSV, and the two NormLens JSON arrays, verifying their archived hashes. NormLens metadata requires downloading its approximately 397 MB source archive. Respect each source's terms when restoring data. Models and feature caches are generated locally; Git ignores them.

After restoring media, `scripts/09_human_packet.py` generates the reusable local annotation interface and `scripts/select_cases.py` generates qualitative assets. Human scoring itself does not require media or the generated interface. Do not expose the answer key to new annotators before completion.

Compile with Tectonic from `latex/`: `tectonic analysis1_preview.tex`, `tectonic human_exercise.tex`, `tectonic intro_discussion_disclosure.tex`; compile `supplement.tex` after generating local qualitative figures. The supplement is five pages; the standalone human record is supporting material, not an uncounted addition to the assignment appendix. Final team integration must respect overall page limits.

## Provenance and disclosure

Pinned source revisions, model revisions, raw-data hashes, and software versions are in [qa/provenance.json](qa/provenance.json). EgoNormia: [source](https://github.com/Open-Social-World/EgoNormia); NormBank: [source](https://github.com/SALT-NLP/normbank); NormLens: [source](https://github.com/wade3han/normlens). The checked VideoNorms source is [author repository](https://github.com/nikhilreddy3/VideoNorms).

AI assistance covered data retrieval, coding, analysis, writing, and checking, as detailed in [sections/ai_disclosure.tex](sections/ai_disclosure.tex). Human responses are participant-supplied; evidence codes are separate post-hoc AI annotations. This folder does not certify other teammates' analyses or the full team submission.
