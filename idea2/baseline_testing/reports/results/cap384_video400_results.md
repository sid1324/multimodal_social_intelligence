# Measured results

All numbers below are computed from saved predictions. Failures remain in denominators.

| Condition | Attempted/panel | Action | Justification | Joint | Sensibility IoU | Invalid examples |
|---|---:|---:|---:|---:|---:|---:|
| base_ego_cap384_video400:video | 400/400 | 48.2 | 40.8 | 30.8 | 48.4 | 48 |
| rloo_ego_cap384_video400:video | 400/400 | 48.5 | 43.0 | 33.5 | 48.9 | 42 |
| sft_ego_cap384_video400:video | 400/400 | 50.0 | 41.2 | 35.0 | 48.7 | 41 |

Scores are percentages. Per-condition tables may have different coverage while jobs run;
use paired_common_panel in summary.json for matched comparisons. Cluster-bootstrap intervals
describe example sampling uncertainty, not variability across training seeds.
