# GPT agent failure diagnostic

RGB baseline: 11/15 jointly correct; action 11/15, justification 11/15.

Follow-up cohort: four selected RGB failures. Not an unbiased benchmark score.

| Condition | Joint recovery |
|---|---|
| pose_images | 1/4 |
| segmentation_images | 1/4 |
| depth_images | 1/4 |
| all_images | 1/4 |
| repeat1_images | 1/4 |
| repeat3_images | 1/4 |
| shuffled_all_images | 1/4 |
| rgb_descriptions | 1/4 |
| pose_descriptions | 1/4 |
| segmentation_descriptions | 1/4 |
| depth_descriptions | 1/4 |
| all_descriptions | 1/4 |
| repeat1_descriptions | 1/4 |
| repeat3_descriptions | 1/4 |
| shuffled_all_descriptions | 1/4 |

Matched controls:
- all_images vs repeat3_images: 0 improved / 0 worsened / 4 unchanged.
- all_images vs shuffled_all_images: 0 improved / 0 worsened / 4 unchanged.
- all_descriptions vs repeat3_descriptions: 0 improved / 0 worsened / 4 unchanged.
- all_descriptions vs shuffled_all_descriptions: 0 improved / 0 worsened / 4 unchanged.

Context accumulation, repeated viewing and outcome-based selection prevent a clean causal geometry interpretation. AI-generated report; student interpretation pending.
