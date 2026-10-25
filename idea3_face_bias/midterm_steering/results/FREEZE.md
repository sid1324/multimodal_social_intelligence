# FREEZE: Idea 3 steering configurations

Frozen at **2026-10-25 16:14:32 EDT**, approved by Yash Lothe, before any test-split run. The test split is evaluated once with these settings.

Selection rule (fixed before the sweep): for each direction type, the lowest validation S_app subject to (i) clean accuracy >= base - 2 pp and (ii) R_beh >= 0.8 x base R_beh; ties to smaller alpha, then fewer layers. Thresholds: accuracy >= 64.8%, R_beh >= 20.6%.

| Method | Layers | Alpha | Direction | Constraint fallback used |
|---|---|---|---|---|
| naive | 14 to 35 | 1.0 | u_app | False |
| refined | 25 | 1.0 | u_perp | False |

Validation (196 cue items, 289 clean items; %, 95% CI clustered by source video):

| Method | Clean accuracy | S_app | R_beh | S_0 |
|---|---|---|---|---|
| base | 66.8 [61.5, 71.9] | 21.4 [16.1, 27.1] | 25.8 [20.9, 30.7] | 2.0 [0.5, 4.1] |
| constitution | 67.5 [62.2, 72.6] | 16.3 [11.5, 21.2] | 29.8 [24.9, 34.8] | 3.1 [1.0, 5.6] |
| naive_L14-35_a1.0 | 65.4 [60.0, 70.5] | 14.8 [10.2, 19.7] | 20.9 [16.0, 25.9] | 4.6 [2.0, 7.7] |
| refined_L25_a1.0 | 65.7 [60.6, 70.8] | 19.1 [13.8, 24.6] | 24.2 [19.6, 29.1] | 3.6 [1.4, 6.3] |

Directions file SHA-256: `a6f41801d188e99a21a10f0e6c4c358103fd02bc359d09aad7ec3d21a92312a6`
Sweep rows SHA-256: `da1bab76e6bebcd4d96f3cdfae8eff80dc9363d9190af4b6c1dbd09afa9f2457`
