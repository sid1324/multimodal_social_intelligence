# Post-midterm visual normative RL pilot: measured outcome

The fixed 25-update RLOO continuation on NormLens high-agreement images
completed on one V100-32 GB. This is a secondary visual-norm training source
applied **after** the R³-FDT video RLOO checkpoint; no EgoNormia label or
example entered training. The [protocol](NORMATIVE_RL_PROTOCOL.md) records
the design. All EgoNormia results reuse the prior 400-ID development panel
at 1,024 output tokens and remain exploratory.

The public NormLens archive was verified at SHA256
`b92f190bca09b748f97a94b2fe058908a93fa3e4ca0025d90f324cd0ed08eba7`.
The image-disjoint high-agreement split contains 716 train, 123 validation,
and 95 reserved test examples. The continuation visited 82 training examples
from 81 images: 25 mixed-reward updates were accepted, 56 uniform-reward
groups were skipped, and one nonfinite-gradient group was skipped. Mean
sampled correctness across four rollouts per example was 63.4%; peak GPU
memory reserved was 12.82 GB. Training took 1:03:38 in Slurm job `50090170`.
The final adapter is
`/ocean/projects/cis260338p/apatawar/social_norm_rl/runs/rloo_normlens/checkpoint-00025`.

| Metric | R³-FDT RLOO starting adapter | NormLens RL continuation |
|---|---:|---:|
| NormLens validation accuracy, 123 images | 59.3% | 61.8% |
| R³-FDT video accuracy, original 60 questions | 80.0% | 80.0% |
| EgoNormia action accuracy, 400 items | 48.2% | 48.0% |
| EgoNormia justification accuracy, 400 items | 44.0% | 44.5% |
| EgoNormia joint accuracy, 400 items | 33.8% | 33.5% |
| EgoNormia sensibility IoU, 400 items | 49.0% | 48.9% |
| EgoNormia invalid examples | 21 | 21 |

The paired EgoNormia joint difference was −0.25 percentage points (95%
source-video-cluster bootstrap interval −2.3 to +2.0). The action difference
was −0.25 (−2.3 to +1.7) and justification +0.5 (−2.0 to +3.0). The
continuation scored 79.7% on the full 79-question accessible R³-FDT test
manifest; the 80.0% comparison above uses the original 60 questions common
to both runs. The NormLens validation change is a small point gain on 123
images, not a robust demonstration of normative learning.

This pilot does not support a transfer benefit from a short image-based
right/wrong/impossible reward stage. NormLens is closer to social norms than
R³-FDT's descriptive movie questions, but its images and three-way judgment
task still differ from EgoNormia's first-person video action selection and
action-conditioned justification. The experiment cannot isolate the effect
of RL from additional training time because a matched NormLens SFT control
was not run. A longer NormLens run is not the next priority given the flat
EgoNormia result; a source with egocentric observations and action/explanation
targets is more directly aligned.

Raw predictions and model artifacts remain on Ocean. Machine-readable paired
summaries and the retained R³-FDT check are in `reports/results/normlens_*`.
