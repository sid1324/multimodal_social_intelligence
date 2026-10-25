# Measured results

All numbers below are computed from saved predictions. Failures remain in denominators.

| Condition | Attempted/panel | Action | Justification | Joint | Sensibility IoU | Invalid examples |
|---|---:|---:|---:|---:|---:|---:|
| base_ego_video400:video | 400/400 | 44.8 | 36.2 | 23.8 | 46.5 | 167 |
| rloo_ego_video400:video | 400/400 | 45.8 | 38.2 | 26.2 | 46.3 | 165 |
| sft_ego_video400:video | 400/400 | 47.8 | 35.5 | 28.0 | 46.8 | 129 |

Scores are percentages. Per-condition tables may have different coverage while jobs run;
use paired_common_panel in summary.json for matched comparisons. Cluster-bootstrap intervals
describe example sampling uncertainty, not variability across training seeds.
