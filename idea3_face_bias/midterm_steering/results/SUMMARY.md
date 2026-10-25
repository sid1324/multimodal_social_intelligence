# Idea 3 midterm: summary of results

Model: Qwen3-VL-8B-Instruct (pinned revision 0c351dd), bf16 on a Colab A100. Data: EgoNormia (pinned), cue items split by source video into train 587 / validation 196 / test 401 (clean accuracy on 921 / 289 / 622 items). Intervals are 95% bootstrap intervals over source videos; "paired" differences are computed item by item against the base model.

## 1. Gate (train, 300 items, base model)
- Appearance sentences ("appears untrustworthy" vs "appears trustworthy") changed the answer on **25.3%** of items [21.0, 29.6], against **2.0%** [0.7, 3.7] for a neutral sentence vs its paraphrase.
- Behavior sentences changed **31.3%**. Accuracy was 55.1% under the low-trust cue vs 60.6% under the high-trust cue.

## 2. Directions (train)
- Cosine between the appearance and behavior directions: about 0 in layers 0 to 9, rising through 10 to 18, **0.78 to 0.81 in layers 19 to 33**, 0.32 at layer 35. Split-half stability at least 0.93 everywhere.
- Reading: the model mostly uses one shared "trust" direction for looks and behavior in its later layers.

## 3. Validation sweep and freeze (196 items)
- Base: appearance flips 21.4%, behavior flips 25.8%, accuracy 66.8%. Constitution: 16.3%, 29.8%, 67.5%.
- Frozen by the pre-registered rule (results/FREEZE.md): **naive** = remove u_app at layers 14 to 35, alpha 1 (14.8%, 20.9%, 65.4%); **refined** = remove u_perp at layer 25, alpha 1 (19.1%, 24.2%, 65.7%). Neither needed the fallback.

## 4. Test (401 cue items, 622 accuracy items), Table 1
| Method | Acc | S_0 | S_app train wordings | S_app held-out wordings | R_beh (held-out) | R_state | S_face |
|---|---|---|---|---|---|---|---|
| Base | 62.4 | 2.5 | 24.5 | 18.0 | 19.7 | 32.4 | 1.2 |
| Constitution | 61.6 | 2.5 | 18.8 | 13.2 | 18.0 | 30.0 | 3.0 |
| Naive | 61.4 | 2.7 | 17.8 | 12.3 | 18.6 | 30.6 | 2.5 |
| Refined | 62.9 | 1.5 | 23.8 | 16.1 | 21.6 | 32.0 | 0.8 |

Paired differences from base (points):
- Naive: appearance flips **-6.7** [-9.4, -4.1] (held-out **-5.6** [-8.4, -2.9]); behavior -1.1 [-3.6, 1.4]; state -1.9 [-4.7, 0.9]; accuracy -1.0 [-2.9, 1.1].
- Constitution: **-5.7** [-8.8, -2.7] (held-out **-4.7** [-8.0, -1.4]); behavior -1.8 [-4.4, 0.9]; accuracy -0.8 [-2.8, 1.1].
- Refined: -0.8 [-2.8, 1.2] (held-out -1.9 [-3.8, 0.1]); behavior +1.9 [0.0, 3.8]; accuracy +0.5 [-0.8, 1.8].

In plain words: naive steering and the constitution prompt both reduce appearance-based flips by about a quarter, also for wordings never used to build or choose the method, with no clear cost to legitimate behavior or emotion responsiveness or to accuracy. Refined steering does not work, because most of the appearance effect lives in the part of the direction shared with behavior.

## 5. Face photos (OMI, test)
- Swapping a higher-trust for a lower-trust face changed the base answer on 4.0% of items vs 2.7% for two equally trusted faces: **no trust-specific face effect** (S_face 1.2 [-1.1, 3.6]). Face-induced shifts are unrelated to the text trust axis (cosine 0.0; same direction on 50% of items). So the text-to-face transfer test could not show anything, and no method changed S_face.

## 6. Failure analysis (details in failure_cases.md)
1. **Avoidance:** under the low-trust cue the base model picks person-directed actions less (58.1% vs 63.8%) and is less accurate (54.5% vs 60.4%).
2. **Invariance without correctness:** naive steering removes the flip on 43 of 139 sensitive items (96 remain); when it removes a flip, the shared answer is correct only 49% of the time (constitution 61%).
3. **No visual trust channel:** faces do not carry the bias in this setup.

## 7. Check of the batched scoring
A one-prompt-at-a-time rerun of 100 test items (2,600 runs) agreed with the batched runs on 99.3% of answers and reproduced the rates within half a point.
