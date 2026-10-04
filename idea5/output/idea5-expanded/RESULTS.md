# Compiled results - stopped at user request

Inference stopped. No new inference is scheduled. 562/640 planned condition rows were saved; 35 clips have all 16 conditions. The two extra rows on clip 36 are preserved in qwen_results.json and excluded from the matched table. All 40 clips have RGB, pose, segmentation and depth grids with descriptions.

Joint action-and-justification correctness:

| Input | Images only | With descriptions |
|---|---:|---:|
| rgb | 3/35 | 2/35 |
| pose | 3/35 | 2/35 |
| segmentation | 4/35 | 2/35 |
| depth | 4/35 | 2/35 |
| all | 4/35 | 3/35 |
| repeat1 | 3/35 | 2/35 |
| repeat3 | 3/35 | 2/35 |
| shuffled_all | 5/35 | 2/35 |

All-three images recover one RGB failure and introduce no joint regressions on these 35 clips. However shuffled-all images score 5/35, versus 4/35 for aligned maps, so a causal geometry advantage is not demonstrated. The images-only recovery is the climbing-equipment case 54adfe07; shuffled maps also recover it. The shopping case e93fdf27 is recovered only by the combined images-plus-descriptions input; repeated and shuffled controls fail. These are selected illustrative examples, not evidence of general improvement.

The GPT-5.6-sol-configured agent workflow gets 11/15 RGB cases jointly correct. Four new RGB failures were retested: one recovers in all 15 follow-up conditions, including repeated RGB and shuffled maps, following image reinspection. This is a tool-using agent diagnostic with accumulating context, not an official benchmark score or stateless comparison.

All 562 Qwen rows were checked against raw answers, permutations, gold indices, image paths and attached-text manifests. Gallery assets and JavaScript syntax were checked; all four PDF pages were rendered and visually inspected. Geometry/caption truth, student interpretation and TA approval remain unverified.

Files: analysis-preview.pdf, review.html, recoveries.html, gpt56-failures.html, scores.csv, summary.json, verification.json, idea5-expanded-section.tex.
