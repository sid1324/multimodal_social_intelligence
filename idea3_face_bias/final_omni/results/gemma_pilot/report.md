# Pilots (Gemma 4 12B, 300 training scenes, base model)

Hearing check (spoken source remark repeated with both the source word and the trait word, 30 scenes): 0.20 (source word 0.37, trait word 0.27; female 0.29, male 0.08). Target 0.90 not met (recorded, not a stop).

| Format | Cue family | n | Flips | Flips minus floor |
|---|---|---|---|---|
| text | src_looks | 300 | 46.5 [41.5, 51.6] | 37.8 [32.0, 43.8] |
| text | src_record | 300 | 52.2 [47.0, 57.4] | 43.5 [37.5, 49.5] |
| text | src_none | 300 | 51.3 [46.0, 56.8] | 42.7 [36.6, 48.8] |
| text | implicit | 300 | 27.1 [23.5, 30.8] | 18.4 [14.2, 22.8] |
| text | floor (neutral rewording) | 300 | 8.7 [5.7, 11.8] | |
| speech | src_looks | 300 | 26.2 [22.4, 30.3] | 14.5 [9.5, 19.5] |
| speech | src_record | 300 | 28.5 [24.2, 32.9] | 16.8 [11.6, 21.9] |
| speech | src_none | 300 | 29.8 [25.4, 34.4] | 18.2 [13.2, 23.3] |
| speech | implicit | 300 | 22.0 [18.7, 25.3] | 10.3 [6.0, 14.5] |
| speech | floor (neutral rewording) | 300 | 11.7 [8.3, 15.2] | |

| Format | Looks flips minus record flips (paired) |
|---|---|
| text | -5.7 [-9.2, -2.2] |
| speech | -2.3 [-6.1, 1.3] |

## Trust ratings (1 to 7; low cue minus high cue, paired by scene)

| Format | Cue family | Low minus high | Pairs in the expected direction |
|---|---|---|---|
| text | src_looks | -2.12 [-2.31, -1.93] | 2 of 2 |
| text | src_record | -4.40 [-4.64, -4.14] | 2 of 2 |
| text | src_none | -3.53 [-3.79, -3.26] | 2 of 2 |
| text | implicit | 0.89 [0.57, 1.19] | 1 of 4 |
| speech | src_looks | 0.41 [0.32, 0.50] | 0 of 2 |
| speech | src_record | 0.49 [0.39, 0.59] | 0 of 2 |
| speech | src_none | -1.13 [-1.29, -0.97] | 2 of 2 |
| speech | implicit | -1.41 [-1.56, -1.26] | 4 of 4 |

Mean probability on the digits 1 to 7: typed 0.985, spoken 0.956.
Agreement between the two scale directions (correlation): typed 0.6971540675491831, spoken 0.6644002755718844. Low agreement means the ratings are not trustworthy.

## Frozen rules (PILOT_PLAN.md)

**Pilot 1 (source control):**
- (b) record above floor, typed: **yes**
- (b) record above floor, spoken: **yes**
- (c) looks above floor, typed: **yes**
- Looks above floor, spoken (decides whether speech has a looks effect to fix): **yes**
- **Pilot 1: GO, the looks vs record contrast becomes the main measure**

**Pilot 2 (implicit, exploratory):**
- manipulation check (typed): low minus high <= -1.0, range below 0, >= 3 of 4 pairs: **no**
- action effect (typed): implicit flips above floor: **yes**
- **Pilot 2: one line: no implicit appearance effect**

