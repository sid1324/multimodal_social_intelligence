# Does Fairness Alignment Cross Modalities?
### Appearance Cues in Text, Speech, and Images in Omni Models

Idea 3, Team 23. Status as of Wed Nov 25, 2026.

## The idea in one minute

AI models that see, hear and read can be unfair because of how someone looks. If you tell a model "that person looks untrustworthy", it may choose a different action, even though looks say nothing about what the person did.

Developers try to train this away, usually with text. **Our question: if we fix the bias using typed remarks only, does the fix also work when the same remark is spoken, or written in the image?** If not, a model can look fair in text and still be unfair in other formats.

Our rule for a fair model: **ignore appearance, but still react to behavior** (for example, someone stealing a wallet) and visible emotions.

## How we test it

1. **Show a scene.** A first-person video scene from EgoNormia (five frames) with four possible actions. The model picks one.
2. **Add a remark** about another person. It is typed, spoken, or shown as a caption under the image.
3. **Measure bias.** We compare answers when the remark says "untrustworthy" and when it says "trustworthy". If the answer changes, that is an appearance flip. We subtract a noise floor: how often the answer changes when a neutral remark is only reworded.
4. **Fix and retest.** We train the model on text examples to ignore appearance, then test all three formats.

Remarks come in three strengths: **L1** a plain sentence, **L2** the same cue inside a realistic description (our main level), **L3** a pushy bystander (used only as a stress test, never in training).

## The fixes, explained simply

**The idea behind both fixes:** remove the effect of looks, keep the effect of behavior. Then check whether a fix made only from **text** also works for **speech** and **images**.

### Fix 1: Steering (no training)

- **What it is.** Inside the model, each remark creates a pattern of numbers. We find the pattern for "looks untrustworthy" by comparing the model's numbers for "untrustworthy" remarks with those for "trustworthy" ones, and averaging over many scenes. We call it a **direction**.
- **How it is used.** While the model answers, we remove that direction, as if turning down one knob. Nothing in the model is retrained, so it can be switched off.
- **Why it might cross formats.** We build the direction from **typed** remarks only. If text and images share the same internal pattern (similarity 0.89 in our data), turning it down should help in images too. **That is a prediction we have not tested yet.**
- **What could go wrong.** Appearance and behavior look alike inside the model (similarity about 0.8), so removing one can weaken the other. We check that the model still reacts to behavior.

| | Typed | Spoken | Caption in image |
|---|---|---|---|
| Direction built from | **yes** | no | no |
| Direction applied to | yes | yes | yes |

### Fix 2: DPO training (a small change to the model)

- **What it is.** DPO (direct preference optimization) shows the model pairs of answers and teaches it to prefer one. We train only a small add-on (LoRA), not the whole model.
- **Two kinds of pairs, made from training scenes:**
  - **Ignore looks (1,408 pairs).** If a remark about looks changes the answer, we teach the model to prefer the answer it gave with no remark.
  - **Behavior still counts (1,036 pairs).** If a remark about behavior (for example, taking a wallet) changes the answer, we teach it to keep that change.
- **Why both.** Training only "ignore looks" could make the model ignore everything.
- **Not used in training:** the pushy L3 remarks and the held-out wordings, so we can test on new phrasing.

### Two versions, and why we compare them

| Version | Trained on | Question it answers |
|---|---|---|
| **DPO-text** | Typed remarks only | Does a text-only fix also reduce bias in speech and images? |
| **DPO-multi** | The same pairs typed, spoken, and shown in the image | What does a fix that sees every format achieve? |

**How we read the result**

| If DPO-text lowers bias in... | Then |
|---|---|
| Text only, not images or speech | Text-only fixing does not transfer. Models could look fair in text and still be biased elsewhere. DPO-multi would be the needed fix. |
| Text and images | The fix carries over between formats, which fits the shared internal pattern. |
| Nothing | The fix did not work, or there was too little bias to remove. |

**Speech is special for Nemotron:** it showed no appearance bias at L2, so there is nothing to fix and we can only report that.

**How we decide honestly.** We pick settings on validation scenes, write them down, then run the test scenes once. A fix counts only if it lowers the bias, keeps accuracy (within 2 points), and keeps at least 80% of the reaction to behavior.

## Midterm results (Qwen3-VL 8B, text remarks)

**Yes, mostly positive.**

| Result | Numbers |
|---|---|
| The bias is real | Appearance remarks changed 25.3% of answers on the training check (noise 2.0%), and 24.5% on test (noise 2.5%) |
| **Steering reduces it** (nudging one internal pattern) | Appearance flips down **6.7 points [4.1, 9.4]** on test; still down 5.6 [2.9, 8.4] for wordings never used to build it |
| **A written rule (constitution) reduces it** | Down **5.7 points [2.7, 8.8]** |
| No clear cost | Accuracy and reactions to behavior did not change significantly |
| Appearance and behavior look alike inside the model | The internal patterns had a similarity of about 0.8, which is why a more precise "surgical" steering did not work (change -0.8, not significant) |
| Faces | High-trust versus low-trust faces changed answers by 1.2 points [-1.1, 3.6], so no clear face effect |

Two limits we found at midterm: both fixes removed about a quarter of the flips, not all. When steering removed a flip, the shared answer was correct only 49% of the time (61% for the constitution).

## What we have now (Nemotron 3 Nano Omni 30B, training split, 300 scenes)

| Question | Answer |
|---|---|
| Is there bias with a plain typed remark (L1)? | **Yes.** 18.0% of answers change, against 4.3% noise (+13.7 points [9.8, 17.7]) |
| At the realistic level (L2), typed? | Yes, smaller. 10.6% against 6.0% (+4.6 [1.6, 7.5]) |
| L2, caption in the image? | Yes. 13.4% against 8.0% (+5.4 [1.7, 9.0]) |
| L2, spoken? | No effect found. 5.4% against 5.3% (+0.1 [-2.9, 2.9]) |
| Does the model really use the picture? | **Yes.** Accuracy 56.0% with the real scene, 29.0% with a blank image, 29.7% with a wrong scene (chance is 25%) |
| Is the bias stored the same way across formats? | **Looks similar.** The internal appearance patterns for text and image have similarity 0.89, text and speech 0.77 (a weather control is only 0.36) |
| Does it hear and read the remark? | Yes, 90 of 90 in a check |

**In plain words:** remarks about looks change what the model decides. The bias comes from the remark, not from judging a face. With a blank image it is still there.

## What we have built

- 2,444 training pairs from training scenes only (1,408 that teach "ignore looks" and 1,036 that teach "behavior still counts").
- Training code with checks on gradients and on resuming after an interruption. All pass.
- Our test scenes are locked until we freeze our choices.

## What is not done

- **No fix has been trained on the new models yet.** Compute is the blocker: the 30B model needs two GPUs and the cluster queue has been very slow.
- **Steering on Nemotron:** 3 of 10 settings tried, no visible improvement yet.
- **At the realistic level (L2)**, validation data shows little bias to remove (2.4 points [-1.6, 6.0]). At L1 there is plenty (13.3 points [9.0, 17.5]), so a fix should show there first.
- Second model and VideoNorms (real TV clips, real dialogue) have not started.

## Next

1. Train on text only, then on all formats (DPO), and compare. Plan: a smaller 2026 open model that fits the GPU we can get (Gemma 4 12B, not yet confirmed).
2. Select settings on validation, freeze, then run the test once.
3. VideoNorms if time allows. Poster and report are due Tuesday Dec 1.

**Open question we will be able to answer:** does a text-only fix lower bias in images? We do not know yet.

## Small glossary

- **Omni model:** one model that takes text, images and audio.
- **Noise floor:** how often answers change just from rewording. Real bias must be above it.
- **Points:** percentage points. "+13.7 points" means 13.7 more answers in 100 changed.
- **[a, b]:** the range we are 95% confident in. If it includes zero, we cannot claim an effect.
- **Steering:** changing one pattern inside the model while it answers.
- **DPO:** training the model to prefer one answer over another.
- **Transfer:** a fix learned in one format working in another.
- **L1, L2, L3:** plain, realistic, and pushy versions of the remark.
