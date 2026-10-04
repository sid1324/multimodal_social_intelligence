# Social Norm Reasoning in Vision-Language Models

CMU 11-777 Multimodal Machine Learning, Fall 2026 — Team 23

Siddharth Singh, Anisha Reddy, Atharv Patawar, Yash Lothe, Shiv Gupta

## Project

We study how vision-language models (VLMs) reason about social norms in egocentric video. Our shared testbed is [EgoNormia](https://github.com/open-social-world/EgoNormia): 1,853 items, each pairing a short first-person clip with candidate actions and the norm categories (e.g. cooperation, safety, proxemics, privacy) that the correct action involves.

Our research ideas build on this benchmark:

- a curriculum from textual norms to visual and video grounding;
- evidence-grounded reinforcement learning for social reasoning;
- whether a person's apparent trustworthiness biases a model's normative choices, and how constitutional preference optimization could remove that bias;
- explicit geometry and body/hand pose as additional inputs for norm reasoning;
- norm-conditioned gating between video and audio evidence.

For Milestone 2, each idea's premises were tested directly on the data, using released model predictions, small human studies, model probes and frozen-model pilots, before any training.

## Repository layout

Each analysis lives in its own top-level folder with its own README describing its code, results and how to obtain the data.

- [Idea 1: Progressive perceptual grounding](idea1/) — rechecked analyses, report sections, and the 20-item human exercise.
- [Idea 2: Evidence-grounded reward feasibility](idea%202/) — Social Genome single-annotator audit, language diagnostics, scripts, tests, and figures.
- [Idea 3: Face-bias premise tests](idea3_face_bias/) — existing Analysis 2 contribution.

## Notes

- Datasets (EgoNormia, Ego4D) are not redistributed here; each folder documents how to download what it needs.
- API keys and local credentials (e.g. `.env`) are never committed.
