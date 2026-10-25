# 01. Project Brief: Idea 3

**Working title:** Beyond the Face: Cross-Modal Mitigation of Physiognomic Bias in Grounded Social Decisions
**Owner:** Yash Lothe (solo idea within the 5-person Team 23 project, CMU 11-777, Fall 2026)
**Team topic:** Social norm reasoning in vision-language models, on the EgoNormia benchmark

## The question in one paragraph
When a vision-language model (VLM) chooses the socially appropriate next action in a real egocentric scene, does the other person's **apparent** trustworthiness sway its choice, even when the correct action doesn't depend on it? If so, can we remove that influence from inside the model **without** also removing legitimate social judgment, such as wariness toward someone who is holding a knife?

## The normative rule (the core of the project)
- **Not legitimate:** judging a person's *character traits* (trustworthiness, dominance, competence) from **what their face looks like**. Facial morphology is not valid evidence of character, yet humans and frontier models read character from faces and act on it. This is the finding of Lehr, Lothe & Banaji (2026, *PNAS Nexus*, "Like humans, language models demonstrate face-to-character biases"), which Yash co-authored.
- **Legitimate:** reacting to what people **do** (behavior and actions, like holding a knife or taking a wallet) and to **states they express** (looking upset or cheerful). These are real social signals.

A good fix removes the first and keeps the second. A model that can no longer be wary of someone pointing a knife is not "debiased," it is broken.

## Why it's multimodal
The same trait impression can reach the model through two channels:
- **Language:** a sentence such as "The other person appears untrustworthy."
- **Vision:** a real face visible in the egocentric frames.

The central multimodal question is whether these converge on **one shared internal "trust" representation**. We test it directly: a direction found from **text alone** is removed, and we check whether that also cuts the influence of **real faces**.

## Research questions
- **RQ1 (does it exist in grounded context?)** Does an apparent-trait cue change Qwen3-VL-8B's normative action choice on EgoNormia, beyond wording noise?
- **RQ2 (is it one concept across modalities?)** Do the face cue and the text cue push the answer in the same direction? Does a text-derived trust direction also mediate face effects?
- **RQ3 (can we remove it selectively?)** Does ablating the appearance-trust direction reduce appearance sensitivity while preserving (a) clean task accuracy and (b) responsiveness to legitimate behavior and expressed-state cues?
- **Sub-question:** How similar are the internal directions for appearance-based trust and behavior-based trust? If they are nearly identical, the model has one trust concept and appearance just feeds into it, and no unconditional removal can be both fair and safe. That motivates the DPO follow-up (see below).

## Method (midterm baseline): activation steering by directional ablation
1. **Directions.** On training scenes with no usable face, run each item twice, with a low-trust and a high-trust *appearance* sentence. Record residual-stream activations at each decoder layer (last prompt token). The mean difference is the appearance direction `d_app[l]`. Do the same with *behavior*-based trust sentences to get `d_beh[l]`.
2. **Variants.**
   - *Naive:* remove `d_app` everywhere.
   - *Refined:* remove only the part of `d_app` not shared with `d_beh`, so `d_perp = d_app - (d_app . u_beh) u_beh`, normalized.
   - *Stretch (position-restricted):* remove the direction only at the tokens where the appearance cue enters (the cue sentence's tokens in the text arm, the face's image-patch tokens in the face arm).
3. **Intervention.** At chosen layers: `h <- h - alpha * (h . u) u`, where `u` is the unit direction. The layer set and `alpha` are chosen on validation with a pre-registered rule.
4. **Comparison baselines:** the base model, and the base model with a **constitution** in the system prompt (no internal change).

## What comes after the midterm (context only, do NOT build now)
- **DPO with invariance preference pairs:** y+ is the model's no-cue answer, y- is its cue-induced answer, trained with LoRA on Qwen3-VL. The pairs are "constitution-specified": the constitution decides which cue types get invariance pairs.
- **The final report compares DPO vs. steering.** Steering deletes a representation *unconditionally*. DPO can learn a *conditional* rule (ignore trust whose source is appearance, keep trust whose source is evidence). Which one works better, and which transfers across modalities?

## Formal setup (use this notation consistently)
- An item is `x = (V, q, O)`: frames `V` (5 egocentric frames), the question `q`, and the option set `O = {o_1..o_K}` (K <= 5, non-empty options only). The gold action is `o*`.
- A cue `c` is drawn from: none, neutral, appearance-low/high, behavior-neg/pos, state-neg/pos. It is delivered as text `c_T` (a sentence) or vision `c_V` (a face visible or blurred).
- The model gives `p_theta(o | x, c)` over the options, from the next-token log-probs of the option letters. Its answer is `y_hat(x, c) = argmax_o p_theta(o | x, c)`.
- **Appearance sensitivity:** `S_app = P[y_hat(x, app_low) != y_hat(x, app_high)]`. Also the continuous version, `E| log p(o*|x,low) - log p(o*|x,high) |`.
- **Noise floor:** `S_0 = P[y_hat(x, neutral) != y_hat(x, neutral_paraphrase)]`.
- **Legitimate responsiveness:** `R_beh = P[y_hat(x, beh_neg) != y_hat(x, beh_pos)]`, and likewise `R_state`.
- **Goal of an intervention `f`:** minimize `S_app(f)` subject to `Acc_clean(f) >= Acc_clean(base) - 2pp` and `R_beh(f) >= 0.8 * R_beh(base)`.
- **Cross-modal alignment (face arm):** per item, the text trust axis is `a(x) = log p(.|x, app_low) - log p(.|x, app_high)` (a K-vector). The face effect is `f(x) = log p(.|x, face_blurred) - log p(.|x, face_visible)`, measured relative to the background-blur control. RQ2 asks whether `f(x) . a(x)` correlates with the face's rated (un)trustworthiness, and whether ablation shrinks it.

## Key references (in the proposal's bibliography unless marked new)
EgoNormia (Rezaei et al., 2025); Ego4D (Grauman et al., 2022); Lehr, Lothe & Banaji (2026); Oosterhof & Todorov (2008); Peterson et al. (2022, OMI); Sobieszek et al. (2026); Ratzlaff et al. (2024, debiasing VLMs by ablating protected-attribute representations); Howard et al. (2024); Chen et al. (2026); Hausladen et al. (2025); Wolfe et al. (2024); Qwen3-VL technical report (Bai et al., 2025). New for the midterm, with titles and venues to be **verified before citing**: contrastive activation addition / representation engineering (e.g. Rimsky et al. 2024; Zou et al. 2023), refusal as a single direction (Arditi et al. 2024), DPO (Rafailov et al. 2023), Constitutional AI (Bai et al. 2022).
