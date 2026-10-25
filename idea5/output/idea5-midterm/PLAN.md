# Idea 5 midterm plan — "Does better spatial information improve social-norm reasoning?"

Written Sat 2026-10-24 16:45 ET (v2: no Gemini API key; Qwen3-VL-Embedding-8B local; Gemini 4 Argon descriptions via `agy`). Deadline Sun 2026-10-25 20:00 ET; writing starts Sun 13:00 at the latest. Paths relative to `~/Desktop/CMU/sem3/multimodal`. Nothing here has been run.

## 0. The claim and how it is decided (fixed now, before any result)

Per input condition k (§3), on the same items: **N_k** = joint norm accuracy (action AND justification) of a small scorer that reads only one embedding vector of the input; **S_k** = spatial-probe accuracy, i.e. whether a linear probe on that same vector recovers geometry facts derived from the maps/landmarks (§6); and controls with the same input budget but wrong or empty spatial content.

Primary contrast (pre-registered): **C5 (all spatial inputs) − C2 (RGB + RGB-only spatial description)**, joint accuracy on the 200 verified items, paired over items. Secondary: C4−C2 (what the mask *descriptions* add given the RGB description — the key contrast because mask descriptions were generated with RGB visible), C3−C2 (mask images), C5−X1 (mismatched masks), C5−X2 (length-matched filler), C5−X3 (repeated RGB).

| Verdict | Rule |
|---|---|
| **Yes** | S_C5 > S_C2 (CI excludes 0) AND C5−C2 CI excludes 0 AND C5−X1 CI excludes 0 (gain requires the *right* spatial content, not more input). |
| **Only for X** | Overall fails, but for taxonomy category X the C5−C2 CI excludes 0 after Holm correction over 7 categories AND it replicates in the grouped-CV set (§7). |
| **No** | S rises (spatial facts demonstrably became readable from the vector) but the C5−C2 upper CI bound < +5 points on both the verified set and the CV set. |
| **Inconclusive** | anything else; report CI and minimal detectable effect (≈7 pts at n=200, ≈4.5 pts at n=600 CV). |

Why this design: Analysis 3 showed appending maps to frozen VLMs gives +1.1 [−1.7, 4.2]. This baseline instead puts the spatial information into a *representation* (one universal multimodal embedding per input) and lets a scorer *learn* from it, so "is the spatial signal present?" (probe) and "does a learner benefit from it?" (scorer) are measured on the same vector. The cleaner causal design (intervene on spatial ability, verify on a spatial benchmark, then measure social accuracy) is §9: a pilot only if a GPU appears tonight, otherwise the final-report plan.

## 1. Data, splits, leakage

* `data/EgoNormia/annotations/final_data.json`: 1,853 items, 1,075 source videos (id prefix before `_`), up to 14 items/video; every item has an empty 5th option, correct in 21 (1.1%).
* **Test T** = the 200-item verified split (183 source videos). Gold-option categories in T: Cooperation 94, Proxemics 75, Politeness 73, Communication 65, Safety 43, Coordination 38, Privacy 15 (multi-label).
* **Train R** = **400** non-verified items, one per source video, source videos disjoint from T, chosen by hash order (`clean60_prepare.py::digest` recipe) before any image or label is read. 400 (not all 1,653) is set by the description and embedding budgets below; the scorer has ~8k parameters so 400 items × 10 options is enough. If descriptions run faster than budgeted, extend R to 600 by the same hash order.
* Never used as input: the `desc` field of `final_data.json` (dataset narrative; can name the action). Quote only in failure analysis.
* Description calls see image files only — no options, question, taxonomy — and run from an empty temp directory holding only that item's strips. Answer key copied to `output/idea5-midterm/private/` which is never a working directory of any call.

Step 1: `scripts/midterm_splits.py` → `output/idea5-midterm/splits.json` `{"test":[200 ids],"train":[400 ids],"train_ext":[next 200],"seed":"midterm-v1"}`. 10 min, no dependencies.

Prep status (measured 16:27): maps 1,127/1,853 and pose 1,057/1,853 after 5 min, both finish ≈16:35; check `prepare_failed.json` / `pose_failed.json` afterwards (Step 0).

## 2. Spatial descriptions — Gemini 4 Argon via `agy` (GPT-6 fallback)

Two calls per item, student's decision that mask descriptions are generated with the RGB strip visible:

* **Call A (RGB + 3 masks → JSON `{"depth","segmentation","pose"}`)**: ~80–120 words each, *spatial content only*: layout; number and placement of people; distance of each person/object from the camera wearer (within arm's reach / near / far); relative positions and facing/body orientation; what is within reach of the wearer's hands; occlusion; motion across the five frames. The prompt states what each map is (depth: brighter = nearer, relative, can be wrong; segmentation: coloured COCO instance masks with labels, black = unsegmented; pose: yellow body skeletons, magenta hands, ≤4 people) and forbids guessing intentions or what should happen next.
* **Call B (RGB only → `{"rgb"}`)**: same spatial prompt style, ~120–150 words. This is the control that makes C4−C2 interpretable: the RGB spatial description is on both sides, so the difference is what the masks added.

Mechanics: `scripts/midterm_describe.py --model gemini-4-argon-<tier> --workers W --split test|train`. Per call: `tmp/<id>/` with only the needed strips; `agy -p --model ... --sandbox --json-schema <schema> --output-format json` from that dir, prompt names the files; parse, retry ×3; write `output/idea5-midterm/descriptions/{id}.json` = `{"rgb","depth","segmentation","pose","describer","seconds"}`; raw replies in `descriptions_raw/`. Resumable.

Measured: low 19 s, medium 40 s, high 120 s per Call A. Budget for 600 items × 2 = 1,200 calls (Call B is cheaper, assume 0.7× Call A):

| tier | W=2 | W=4 | W=6 |
|---|---|---|---|
| low (19 s) | 2.8 h | 1.4 h | 0.95 h |
| medium (40 s) | 5.7 h | 2.8 h | 1.9 h |
| high (120 s) | 17 h | 8.5 h | 5.7 h |

**Rule**: high is out. Use **medium if the concurrency test shows W ≥ 4 sustained, otherwise low**. Decide before starting T, and use one tier/describer for T and R (train/test must share the describer). Order: T first (400 calls), then R, then `train_ext` if time remains. Expected: start 17:15, T done ≈18:30, R done ≈21:00 (medium, W=4).
Fallback: `codex exec -m gpt-6-sol -i <rgb> -i <depth> -i <seg> -i <pose>` (~12 s/call; 6 workers gave ~400 calls/h on 10-04) — if adopted, redo T with it too. Last resort: local Qwen3-VL-8B-4bit (mlx_vlm, ~3 s/call, GPU-exclusive).
Audit (Step 2b, Sun morning, 20 min): hand-check 20 T items; count hallucinated people/objects/relations per map type; report in failure analysis.

## 3. Embedder, condition ladder, controls

**Embedder: `Qwen/Qwen3-VL-Embedding-8B`** (Apache-2.0; 4096-d; 32k context; MMEB-V2 top; text+images in one sequence → one vector). Verified from the repo's `src/models/qwen3_vl_embedding.py`: `Qwen3VLEmbedder(model_name_or_path, min_pixels, max_pixels)` loads a custom `Qwen3VLForEmbedding` (trust_remote_code) + `Qwen3VLProcessor(padding_side='right')`; `process(inputs: List[Dict], normalize=True)` takes dicts with `text` / `image` (single or **list**) / `video` / `instruction`; `format_model_input` appends videos, then images, then text into one chat message (not interleaved, which is fine: one sequence, one vector); pooling = last hidden state at the last non-pad token, L2-normalised; **no MRL truncation in code** (truncate + renormalise ourselves if we want a 1024-d ablation). Pins: `torch==2.8.*`, `transformers>=4.57.3`, `qwen-vl-utils>=0.0.14`; our venvs have torch 2.14 / transformers 5.18, so make a fresh **`.venv-embed`** with the pinned versions (Python ≥3.11) and run on MPS in bf16 (≈16 GB weights; 128 GB is ample). Instruction: fixed string `"Represent this egocentric scene for choosing the socially appropriate next action."` for contexts; `"Represent this candidate action."` / `"... candidate justification."` for options. MPS caveats: no flash-attn (use `attn_implementation="sdpa"`), `device_map` not used, and if the custom class fails under 2.8/MPS the fallback is the `sentence-transformers` path shown on the model card (`SentenceTransformer("Qwen/Qwen3-VL-Embedding-8B")`, dict inputs `{"text":..., "image":...}`), and the **2B** variant (`Qwen/Qwen3-VL-Embedding-2B`, 2048-d) if 8B is too slow (§11 budget).

Each condition yields one vector per item. Image parts are the 5-frame strips (1920×384 → ≈720 visual tokens each at 32 px/token; cap `max_pixels` so one strip ≤ 800 tokens; four strips + ~900 text tokens ≈ 4k tokens per forward).

| k | Context parts | Tests | Needs descriptions? |
|---|---|---|---|
| C0 | fixed null string | option-prior / text-only baseline | no |
| C1 | RGB strip | raw pixels | no |
| C2 | RGB strip + D_rgb | pixels + RGB spatial description (**reference**) | yes |
| C3 | C2 + depth, seg, pose strips | mask images add? | yes |
| C4 | C2 + D_depth, D_seg, D_pose | mask descriptions add? | yes |
| C5 | C2 + mask strips + mask descriptions | everything | yes |
| C6 | D_depth, D_seg, D_pose only (text) | spatial text alone | yes |
| X1 | C5 with masks AND mask descriptions from another item (other source video, fixed hash pairing) | wrong spatial content, same budget | yes |
| X2 | C2 + fixed non-spatial passage cut to the token length of the item's three mask descriptions | more tokens, no information | yes |
| X3 | C2 + RGB strip ×3 more (4 images, as C3) | more image tokens, no new information | yes |

Options: each of the 10 option texts embedded separately (text-only, batched — batching is fine here, nothing is generated).

Execution order so partial results still answer the question: options, C0, C1, X3 (no descriptions needed; start as soon as the embedder runs) → C2, C5, C4, X1 → C3, C6, X2. Script `scripts/midterm_embed.py --cond ... --split ...` → `output/idea5-midterm/emb/{cond}.npy` (rows in `splits.json` order), `emb/options_action.npy`, `emb/options_just.npy` (n×5×4096), `emb/manifest.json` (token counts, seconds, errors). Resumable per (item, cond); fp16 on disk.

Budget (600 items): 6 image conditions × 600 = 3,600 image forwards; 4 text conditions × 600 + 6,000 option texts, fast. The Step 3a timing probe (3 items, 4 strips) sets the plan: at ≤5 s/forward → ≤5 h, run overnight; 5–8 s → drop C6/X2 first, then reduce to R=300; >8 s → switch to the 2B model (≈3× faster) and say so. GPU-exclusive: nothing else on MPS while this runs.

## 4. From one vector to an answer: scorer and loss

Notation for the report: item i, condition k: context vector c_i^k ∈ R^D; action options a_{i,1..5}, justification options j_{i,1..5} with embeddings o ∈ R^D; gold y_i^a = y_i^j. Joint accuracy 1[ŷ^a = y^a ∧ ŷ^j = y^j].

**Zero-shot (Z)**: ŷ = argmax_m cos(c_i, o_{i,m}) over the four non-empty options. Free; shows how much the embedding alone "knows"; expected weak because cosine rewards the option that *describes* the scene.

**Trained scorer (primary)** — diagonal bilinear, 2D+1 ≈ 8k parameters, identical for every condition:

  s_θ(c, o) = Σ_d w_d c_d o_d + v·o + β·1[o is the empty option]
  p(m | i) = softmax_m s_θ(c_i, o_{i,m})
  L(θ) = −(1/|R|) Σ_{i∈R} [ log p(y_i^a | actions) + log p(y_i^j | justifications) ] + λ(‖w‖² + ‖v‖²)

Prediction: independent argmax per head. Under C0, c is constant so s reduces to v·o — the option-prior baseline falls out of the same model. λ and epochs by 5-fold CV inside R only; full-batch Adam, 10 seeds, majority vote over seeds for the reported predictions, seed spread reported. Pre-committed second architecture if validation shows underfitting: rank-32 bilinear (Uc)·(Vo). Not an MLP over the concatenation: 400 items cannot support it and the comparison must stay about the *vector*, not capacity.

`scripts/midterm_scorer.py` (`.venv-lang`, torch+sklearn) → `results/{cond}_test_preds.json`, `results/{cond}_cv_preds.json`. Written and unit-tested on random vectors before embeddings exist (Step 4a).

## 5. Statistics

Paired over items. Every contrast: exact McNemar on joint correctness + 95% percentile bootstrap (10,000 resamples; for CV resample source videos) of the accuracy difference; fixed/broken counts. One primary contrast uncorrected; 5 secondary Holm-corrected; 7 category contrasts Holm-corrected separately (category = taxonomy of the gold option, multi-label, stated). Also action-only and justification-only accuracy, Z row, and the 60 Analysis-3 items as a sub-row of T. `scripts/midterm_stats.py` → `results/stats.json`, `table_main.tex`, `table_categories.tex`, `fig_ladder.pdf` (ladder with CIs, controls hollow), `fig_probe_vs_norm.pdf`.

## 6. Spatial probe — showing spatial information rose, separately from norm accuracy

Labels per item from extractor outputs only (`scripts/midterm_spatial_labels.py` → `spatial_labels.json`, `.venv-geometry`/`.venv-pose`, 15 min, light GPU):
`n_people` {0,1,2,3+} (max over frames of YOLOv8n person detections re-run on the five tiles of `{id}_rgb.jpg`, cross-checked with MediaPipe pose count); `nearest_dist` {near, mid, far} (terciles of the median depth inside the largest person mask, read from `{id}_depth.jpg`); `facing` {toward, away/side, none} (nose between visible shoulders in `pose_landmarks/{id}.json`); `hands_visible` {0, 1+}.

Probe: multinomial logistic regression on the frozen vector, 5-fold grouped CV over all 600, one per label per condition; S_k = mean balanced accuracy over the four labels; chance from label frequencies.

Circularity, stated plainly in the report: for C3/C5 the labels come from the maps the embedder saw, so S there is a **manipulation check** ("the facts became linearly readable"), not an independent measure of spatial intelligence. The informative numbers are S_C1/S_C2 (geometry the RGB embedding already carries) vs S_C3..C5, and S_X1 (mismatched maps must not raise S on the item's own labels).

Linking S to N: (a) same ordering of conditions by S and N? (b) per item, cross-tab Δprobe-correct vs Δnorm-correct (C5 vs C2), McNemar/odds ratio; (c) per category, scatter ΔS vs ΔN with CIs (Proxemics/Privacy should sit top-right if the hypothesis holds); (d) items whose gold option contains a spatial term (Analysis-3 word list) vs not. "Only-for-X" claims come from (c)/(d).

## 7. Evaluation protocol

* Main table: train on R (400), test on T (200); T never used for selection.
* Power table: 5-fold CV grouped by source video over all 600 (T+R), λ chosen on training folds only; 600 paired predictions per condition. All "only-for-X" claims must replicate here.

## 8. Optional frozen-VLM arm (only if the GPU is idle, lowest priority)

Qwen3-VL-32B-8bit on T, one stateless call per item (`clean60_qwen.py` recipe), conditions {text, RGB, RGB+D_rgb, RGB+D_rgb+D_maps, RGB+maps, mismatch}: 200×6×~4 s ≈ 1.5 h GPU. Shows whether a frozen VLM uses the *text* descriptions (Analysis 3 only tested map images). Competes with §3 for the GPU; run only after embeddings finish, or on Bridges-2.

## 9. The causal design (intervene → verify spatial gain → measure social change): feasibility verdict

Design: improve spatial ability of Qwen3-VL-8B by LoRA on a spatial dataset; verify on a held-out spatial benchmark; measure EgoNormia T joint accuracy (overall + per category) against a control LoRA of identical recipe and size on non-spatial data; base model as reference. Contrast of interest: (spatial-LoRA − control-LoRA) on T, paired bootstrap; mediation = that contrast is larger on Proxemics/Privacy and on items whose spatial-probe label is relevant.

Datasets checked (read-only, 16:50): **SAT** (`array/SAT`, HF, MIT; 172k train / 127k static / 4k synthetic val / 150 real test; fields `image_bytes` (1–2 images), `question`, `answers`, `correct_answer`; `load_dataset("array/SAT")`, parquet 7.6 GB total; `array/SAT-v2` larger; `array/SAT_perspective` 6.5k perspective-taking) — **downloadable, the practical choice**; held-out spatial check = SAT val (synthetic) + SAT test (150 real) + our §6 labels on EgoNormia frames (in-domain, noisy). **EgoProx** (CVPR 2026, 2,405 VQA from ADT/EgoExo4D): no public download found. **Ego3D-Bench** (ICLR 2026, 8.6k QA, outdoor multi-view): no HF dataset found, release unclear. **Open3DVQA** (`EmbodiedCity/Open3DVQA`, Apache-2.0): aerial/UAV imagery, wrong domain. Control data: equal-size general VQA, e.g. the `vqav2` subset of `HuggingFaceM4/the_cauldron` (verify at download).

Verdict: **not a main midterm step.** It needs LoRA training (`mlx-vlm[train]` is uninstalled and untested with Qwen3-VL image LoRA; HF PEFT needs a CUDA GPU), three model evaluations on T (600 isolated calls) plus SAT evals, and the local GPU is already committed to §3 overnight. **Pilot go/no-go Sat 22:00**: go only if (i) Bridges-2 SSH works and a GPU job starts within 30 min, (ii) descriptions and embeddings are running unattended, and (iii) the §3 timing probe passed. Pilot recipe: 5k SAT (static) vs 5k VQAv2 control, LoRA r=16 on Qwen3-VL-8B, 1 epoch bf16 (≈1–2 h on one A100 with HF PEFT), eval SAT val 500 + EgoNormia T (RGB strip input, `clean60` prompt) for base/spatial/control = 1,500 isolated calls (~1 h on GPU). If no-go, this becomes the **updated research idea** with the math: L = L_norm(θ) + λ L_spatial(θ) where L_spatial is either the SAT cross-entropy or InfoNCE between the RGB-only representation f(V) and the map representation g(depth, seg, pose) (spatial distillation, so no extractors at inference), evaluated with the same three-model contrast and the §3 controls; PSC Bridges-2 is where it runs for the final report.

## 10. Failure analysis (Sun morning, ~1.5 h)

1. Fixed/broken lists for C5 vs C2 and C4 vs C2, with category and spatial-term flags.
2. Six items (3 fixed, 3 broken): RGB + pose strips, the three mask descriptions, gold vs predicted options; does the description contain the spatial fact the norm hinges on (e.g. "within arm's reach" on a Proxemics item)?
3. Describer error rate from §2b and whether errors concentrate in broken items.
4. The 21 "none"-correct items and the β term.
5. Zero-shot cosine's confusion pattern (picks the most scene-descriptive option?) — norm reasoning is not retrieval.

## 11. Report mapping (Idea 5 share: ~1 p baseline, ~1 p results, ~0.5 p updated idea)

* **Problem statement (shared)**: x = (frames V, options {a_m},{j_m}), spatial side-information g(V) = (depth, seg, pose) and its verbalisation, y = (y^a, y^j); objective max_θ E[1[ŷ = y]] with the §4 CE surrogate.
* **Baseline**: "universal multimodal embedding + trained lightweight scorer": Qwen3-VL-Embedding-8B (Jan 2026, MMEB-V2 #1) with spatial side-information rendered as maps and verbalised by Gemini 4 Argon; §4 loss; zero-shot cosine as the untrained reference; figure = the ten-rung ladder.
* **Results**: `table_main.tex` (rows C0–C6, X1–X3, Z; columns joint/action/justification on T, joint on CV-600, probe S), `fig_ladder.pdf`, `table_categories.tex`; one sentence per §0 rule stating the verdict the numbers support.
* **Failure analysis**: §10 with two figures.
* **Updated research idea**: §9 design with its math and the midterm numbers as motivation (if "No": where geometry is needed, conditional routing by category; if "Yes": distil the maps into the RGB model).
* **Disclosure**: Gemini 4 Argon (descriptions), Qwen3-VL-Embedding-8B (embeddings), GPT-6/Qwen only if fallbacks were used, Claude Code and Codex for planning/code/text.

## 12. Steps, dependencies, outputs, timeline (ET)

| # | Step | Depends on | Output | When | Dur. |
|---|---|---|---|---|---|
| 0 | Confirm prep finished; copy answer key to `private/` | running scripts | `prepare_failed.json`, `pose_failed.json` | Sat 16:50 | 5 min |
| 1 | Splits | 0 | `splits.json` | 16:55 | 10 min |
| 2a | Pick describer tier from the concurrency test; `agy` JSON/image sanity on 3 items | 0 | `throughput.json` | 17:00 | 15 min |
| 2 | Descriptions T → R → train_ext (background, remote) | 1, 2a | `descriptions/*.json` | 17:15 → ≈21:00 (worst ≈01:00) | 2–4 h wall |
| 3a | `.venv-embed` + download 8B (+2B) + timing probe (3 items, 4 strips) | 0 (GPU free after prep) | `emb/timing.json` | 17:00–18:30 | 1.5 h (download-bound) |
| 3b | Spatial labels | 0 | `spatial_labels.json` | 18:30 | 15 min |
| 3c | Embed options, C0, C1, X3 (no descriptions needed) | 3a | `emb/*.npy` | 18:45–20:30 | ≈1.5 h GPU |
| 4a | Scorer + stats code on random vectors | 1 | `midterm_scorer.py`, `midterm_stats.py` | 18:30–21:00 | 2.5 h (CPU, parallel to 3c) |
| 3d | Embed C2, C5, C4, X1, then C3, C6, X2 (resumable, unattended) | 2 (T then R), 3c | `emb/*.npy` | 21:00 → ≈03:00 | ≤5 h GPU |
| 9p | LoRA pilot go/no-go | Bridges-2 | — | 22:00 | — |
| 5 | Zero-shot + scorer, T and CV; probes | 3b, 3d | `results/*_preds.json` | Sun 08:30 | 45 min |
| 6 | Stats, tables, figures | 5 | `results/stats.json`, `table_*.tex`, `fig_*.pdf` | 09:15 | 1 h |
| 7 | Failure analysis + description audit | 6 | `failure_cases.md`, 2 figures | 10:15–12:00 | 1.75 h |
| 8 | Optional Qwen VLM arm | 2(T), GPU idle | `results/qwen32b_T.json` | only if 3d ends early | 1.5 h |
| 9 | Write Idea 5 sections (LaTeX, tectonic) | 7 | `report/idea5_*.tex` | 13:00–18:00 | 5 h |
| — | Buffer / team merge | | | 18:00–20:00 | 2 h |

Critical path: prep → descriptions(T) → embeddings(C2/C5/C4/X1 on T) → scorer → stats → write. Minimum viable result if things slip: T only (no R) with 5-fold grouped CV over T's 200 items for conditions C0, C1, C2, C4, C5, X1 — that still yields the primary contrast, the mask-description contrast and the mismatch control, with ±7-pt precision; it needs only 200×2 description calls (~1 h) and 200×4 image forwards (~20 min).

Parallelism rules: descriptions and codex/agy calls are remote and run beside anything; exactly one GPU job at a time (prep → embedder probe → §3c → §3d → optional §8); scorer/stats on CPU alongside the GPU job.

## 13. Files to create (existing files untouched)

`scripts/midterm_splits.py`, `midterm_describe.py`, `midterm_spatial_labels.py`, `midterm_embed.py`, `midterm_scorer.py`, `midterm_stats.py`, `midterm_failures.py`, optional `midterm_qwen_arm.py`, optional `midterm_lora_pilot/` (Bridges-2 only).
`output/idea5-midterm/{splits.json, throughput.json, descriptions/, descriptions_raw/, spatial_labels.json, emb/, results/, private/answer_key.json, failure_cases.md}`; new venv `.venv-embed`; models under `models/Qwen3-VL-Embedding-8B` (and `-2B`).
