# Plan: does text-built steering transfer from text to images? (Qwen3-VL 8B)

Approved by Yash, Nov 25, 2026. Poster and report are due Tue Dec 1.

## The question
At midterm, steering with a direction built from **typed** remarks cut appearance flips by 6.7 points on Qwen3-VL 8B. **Does the same frozen steering also cut bias when the remark is a caption in the image?**

Qwen3-VL has no audio, so speech comes later from an omni model.

## What is already fixed (no new tuning)
- Model: Qwen3-VL-8B-Instruct, the same pinned revision as midterm
- Scenes, cue sentences and train/validation/test splits: the same files as midterm
- Steering: direction `u_app` from midterm, layers 14 to 35, full strength (frozen Oct 25)
- Remark sentences: the midterm ones (L1)

Because nothing is tuned for images, the image result is a clean test of transfer.

## Models, in order (Yash, Nov 25: several small models before telling teammates)
| Order | Model | Why | Speech |
|---|---|---|---|
| 1 | Qwen3-VL 8B | midterm result exists; frozen settings, nothing to tune | no |
| 2 | Gemma 4 12B (`google/gemma-4-12B-it`) | different family, newer, takes text, image and audio | yes |
| 3 | Qwen3.5 9B (`Qwen/Qwen3.5-9B`) | newer; run only if Colab units remain | no |

- Budget: 100 Colab units, A100 40 GB at about 5.5 to 7 units per hour. Estimates: 3 to 4 hours for model 1, 4 to 5 for models 2 and 3. Run one at a time; stop a model that goes over its estimate.
- Report every model run, including failures. Positive means at least 2 of 3 models pass the success rule (2 of 2 if units run out).
- Model 1 uses the frozen midterm direction. Models 2 and 3 build their own direction on training scenes and choose settings on validation with the midterm rule.
- Hand over to Atharv and Shiv only after model 2. After model 1 alone, say "promising, confirming in a second model".

## Success rule, written before any steered image run
Count it as **success** only if ALL hold on the scenes used:
1. **Bias exists:** unsteered image flips minus the image noise floor is above zero (95% range excludes 0).
2. **Steering helps in images:** steered image flips are lower than unsteered, same scenes, range excludes 0.
3. **It is not just any disturbance:** a random direction of the same kind does not give a similar drop.
4. **No damage:** accuracy within 2 points, and at least 80% of the behavior reactions remain.

**Partial:** 1 and 2 hold but 3 or 4 fails. **Not successful:** 2 fails, or 1 fails (then there is nothing to transfer).

## Checklist

### Phase 1. Set up model 1 (Claude, 1 to 2 hours; models 2 and 3 after model 1 passes its checks)
- [ ] 1.1 Copy the midterm Qwen code unchanged into a new folder; add only what is new
- [ ] 1.2 Add the image version: scene with a white strip under it holding the same sentence, plus one line saying a remark is shown as a caption
- [ ] 1.3 Add the random-direction control (3 fixed random directions, same layers and strength)
- [ ] 1.4 Make the Colab notebook: resumable, saves to Drive, stops itself, no test-split run possible without Yash's OK
- [ ] 1.5 CPU checks: prompts, captions, splits, and that the new code gives the same text prompt as midterm
- [x] 1.6 Yash reviews the caption line and the success rule above (approved Nov 25, 22:10 CT)

### Phase 2. Checks on Colab (A100 40 GB, about 30 minutes)
- [ ] 2.1 The model loads and scores letters
- [ ] 2.2 **Reproduce midterm:** text answers on validation match the saved midterm rows (at least 98%)
- [ ] 2.3 **Can it read the caption?** Repeat the remark on 30 scenes, at least 90% correct
- [ ] 2.4 Speed and cost estimate before the long run

### Phase 3. Base bias in images (about 30 minutes)
- [ ] 3.1 Unsteered answers for text and image remarks on validation (196 scenes) and 300 training scenes
- [ ] 3.2 **Gate:** is image bias above its floor? If no, stop and tell Yash (options below)

### Phase 4. The experiment (validation, about 1 to 2 hours)
- [ ] 4.1 Run five methods on text and images: unsteered, constitution, naive steering, and 3 random directions
- [ ] 4.2 Measure per method and format: appearance flips, floor, behavior flips, accuracy, new wordings
- [ ] 4.3 Compare scene by scene with 95% ranges; check the success rule
- [ ] 4.4 Report to Yash with numbers, success or not

### Phase 5. Final test (only after Yash approves)
- [ ] 5.1 Write the freeze file, commit, Yash reviews
- [ ] 5.2 Run the test scenes once (401), text and images
- [ ] 5.3 Table, figure, 3 failure cases, limits

### Phase 6. Hand over to Atharv and Shiv (only if Phase 4 or 5 succeeds)
- [ ] 6.1 One-page spec: the question, exact method, success rule, what not to touch
- [ ] 6.2 Code pack: cue files, splits, caption builder, steering and scoring code, analysis script
- [ ] 6.3 Each runs the same steps on one bigger model: base bias, build direction on training scenes, choose settings on validation, freeze, then test once
- [ ] 6.4 At least one model must hear speech, for the speech arm
- [ ] 6.5 They send back rows in our format; Claude combines the results

### Phase 7. Poster and report
- [ ] 7.1 Main figure: text vs image steering effect, with the random control
- [ ] 7.2 Main table: model by format, with ranges
- [ ] 7.3 Limits stated plainly (caption, not faces; small effects; one dataset)
- [ ] 7.4 Poster done Sun Nov 29, review Mon Nov 30, submit Tue Dec 1

## Timeline (ET, estimates)
| When | What |
|---|---|
| Wed evening | Phase 1 |
| Wed night or Thu morning | Phases 2 to 4 on Colab (Yash presses Run) |
| Thu | Results and decision; freeze if good |
| Fri | Test run; hand over to Atharv and Shiv |
| Sat to Sun | Teammates run their models; figures |
| Mon | Report and poster review |
| Tue Dec 1 | Submit |

Cost on Colab: roughly $3 to $5, not measured yet.

## If it does not work
| What happens | What we do |
|---|---|
| Images show no bias | Say so. Try the realistic L2 remarks once, as an exploratory check, then stop |
| Image bias, but steering does not help | Report "text steering does not transfer to images". Build a direction from image remarks instead, as a follow-up |
| Random direction helps as much | Steering is not specific; we cannot claim the method works |
| Behavior or accuracy drops | Report the trade-off, do not call it a success |

## Delta
DPO and steering jobs on Nemotron stay in line. They cost nothing while waiting (150 SU pause still holds).

## What Yash must do
1. Approve the caption line and the success rule (Phase 1.6)
2. Press Run in Colab, keep the window open, and approve each freeze before a test run
3. Ask Atharv and Shiv now which model and GPU each can use, and by when
4. VideoNorms stays parked until this is done

## Update Nov 25, 23:05 CT: switch to speech on Gemma 4 12B (Yash)
- The caption experiment is dropped as the main result: it tests reading text in an image, not visual bias. Not sent to the friend.
- New main experiment: steering built from typed remarks on Gemma 4 12B, tested on spoken remarks (Kokoro audio, same sentences).
- Cue level, decided before any Gemma result: **L1 main, L2 also reported.**
- One Colab notebook runs everything, GPU and CPU, with automatic stops: checks, bias gate, direction, text sweep with the midterm selection rule, speech transfer with random-direction and constitution controls, report, zip. Test split stays locked until Yash approves a freeze.
- Same 4-part success rule as above, with "image" replaced by "speech".
