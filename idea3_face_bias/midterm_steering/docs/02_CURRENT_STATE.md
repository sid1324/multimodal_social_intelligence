# 02. Current State and Decision Log (as of Sat Oct 24, 2026, 10:30 am ET)

## Course timeline so far
| Milestone | Date | What Idea 3 did |
|---|---|---|
| Preproposal | Sep 2026 | Composite trait-manipulated faces into "blurred" regions; text arm; compare prompt instruction vs. face augmentation |
| Proposal + presentation | ~Sep 20 | Within-identity trait manipulation (Sobieszek vectors); text arm; **mitigation = ablate the trust direction (Ratzlaff et al. 2024) vs. prompt instruction** |
| A2 / Milestone 2 (data analyses) | Oct 4 | Premise tests P1 to P5 (below) |
| **Midterm report** | **Oct 25, 8pm** | **First implementation (baseline) of the idea, test-set results, failure analysis, updated idea** |
| Final report | December | DPO vs. steering comparison, cross-modal generalization |

## Feedback received
**Professor (at the proposal presentation):**
- Do the bias analysis by the midpoint and keep it brief. Spend November on mitigation. It cannot be just an analytical study.
- Emphasize the multimodal part (both arms). The class is modeling/ML-focused, not analysis-focused.
- All ideas must be multimodal by the final.
- He mentioned activation steering because Yash had pitched it, not as a requirement. He is open to ideas changing.

**TA (Kashish, primary TA), relayed on the proposal:** "Comparing predictions with visible versus blurred faces could provide a useful analysis for the midterm. Clarify whether this examines facial information alone or its interaction with other modalities, and consider a modeling approach to reduce any bias you find." The overall feedback was that the proposal needs a clearer account of what changes in the model, how those changes use multiple modalities, and how we test improvement.

## What A2 (Milestone 2) established (the code and results are in `premise_analysis/`)
Data: all 1,853 EgoNormia items, pinned to annotations revision `09d8a7c53f06ed582236722ba8496c6a504cd1ab` and images revision `2937df7fa96d8515417e7b5122fbf8eaeeaeee86`. 1,075 distinct source videos (the id prefix before `_`).
- **P1/P2, faces exist (this overturns the proposal's assumption).** Partner faces are mostly *not* blurred. Of 34 real faces among 60 labeled detections, 26 were unblurred (76%). In 40 uniformly sampled items, 12 (30%, 95% CI [18, 45]) had a usable face, roughly 560 items. Faces are small (median about 50 px in 640-px frames). 26 of 60 detections were not faces at all, so automatic detection needs verification. All labels came from Yash, blind to automatic scores.
- **P3, room for a cue.** 54% of gold answers are person-directed, and 65% of items mix person-directed with other options. Released wrong answers show no pre-existing lean toward person-directed options (within 3.3 pp of uniform for all 8 VLMs).
- **P4, generated face edits are confounded.** OMI reference-guided edits confined pixel changes to the mask (one mask leaked onto a neighbor's glass and hand). But high-trust references **smile** and low ones don't, and donor transfer sometimes changed identity or apparent ethnicity. **Decision: no face generation for the midterm.**
- **P5, the text cue works on a frontier model.** On Gemini 3.1 Pro with 60 items and default sampling, low-trust vs. high-trust prompts gave different answers on 30% of items, against 12% for two identical neutral prompts and 15% for a paraphrase. Excess over the repeat: 18 pp [7, 30], sign test p = 0.007. Accuracy was 60% low, 67% neutral, 72% high. Person-directed choice did not shift.
- A2's committed next steps: use Qwen3-VL as well as a frontier model; scale the text arm to about 200+ items; add an incidental-mention control; manipulate structure with expression held fixed if faces are ever generated.

## Decision log (all approved by Yash)
1. **Midterm baseline = activation steering (directional ablation).** DPO comes after the midterm, and the final compares the two. Rationale: lower risk this weekend, matches the proposal of record, and makes a cleaner story.
2. **Measurement:** greedy decoding with log-probs of the option letters (no sampling). Splits are by source video.
3. **Four test checks:** (1) held-out cue wordings, (2) text-to-face transfer, (3) clean accuracy, (4) legitimate-cue responsiveness (behavior and expressed state).
4. **Visual arm = real faces,** each item shown face-visible, face-blurred (by us) and background-blurred (same-size blur elsewhere). A frontier model rates each face's apparent trustworthiness *and expression* (smiling or not, valence) so expression can be analyzed as a confound.
5. **Blur-elsewhere control** is approved.
6. **Splits:** every source video containing a usable-face item goes to test. Remaining source videos: 50% train (direction extraction), 15% validation (layer and alpha selection), 35% text test.
7. **Framing:** appearance-inferred traits are ignored; behavior and expressed states still count. This is the constitution's core rule.
8. **Ablate (project out), don't add.**
9. **Fix 1 (required):** behavior-based trust cues are a required control and must still change answers after steering.
10. **Fix 2 (required):** the refined direction is orthogonalized against the behavior direction. The appearance/behavior cosine is reported as a finding.
11. **Fix 3 (stretch):** position-restricted ablation, only if the core pipeline is done by about mid-afternoon Saturday. Otherwise it goes in the "updated ideas" section.
12. **Sunday:** no DPO pilot. Writing starts once steering results exist.
13. **Gateway budget:** frontier calls for face verification, face ratings and an optional frontier comparison row are approved in principle. Claude must show the call count before running each batch.
14. **Compute:** the PSC Bridges-2 allocation `cis260338p` is active (2,000 GPU hours and 3 TB exchanged, about 1,969 hours left on Oct 24). V100s are confirmed, A100/H100 availability is unknown. Steering is inference only, so V100-32GB is enough. Model weights go in the team-shared `HF_HOME=/ocean/projects/cis260338p/shared/hf_cache`.
15. **Folder:** all midterm code goes in `idea3_face_bias/midterm_steering/` in the team repo. The rest of `idea3_face_bias/` is the A2 work (read-only).

## What does NOT exist yet
- Any Qwen3-VL inference (A2 used only Gemini via the gateway and rescored released predictions).
- The verified list of usable-face items across all 1,853 (only 60 detections and 40 items were human-labeled).
- Splits, directions, steering code, results.
- EgoNormia images on PSC (they must be re-downloaded at the pinned revision).
