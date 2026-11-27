# Handoff: Idea 3 state for a new Claude Code session (Nov 26, 2026, evening)

Read this first. Team 23, CMU 11-777 (MMML). Poster and report due Tue Dec 1, 2026. Owner: Yash Lothe. Teammates: Shiv, Atharv
(working on this idea); Sid and Anisha (backup ideas). Repo: https://github.com/sid1324/multimodal_social_intelligence, branch
`idea3-final`, folder `idea3_face_bias/final_omni/`.

## Working rules (Yash)
- No test-split runs until Yash approves a freeze. No pushes unless Yash asks. No em dashes in anything written for Yash or the report.
- One-line explanation beside each terminal command. Yash makes research decisions; give honest feedback, do not just agree.
- For contested research calls, use an impartial independent reviewer (subagent) and report its verdict faithfully.
- Poll running jobs and fix problems within approved limits without being asked.

## The question and what we found
Does a fairness fix learned from TYPED remarks also work when the remark is SPOKEN? Model: Gemma 4 12B (google/gemma-4-12B-it,
revision 43c9d014...), EgoNormia egocentric scenes, choose the most socially appropriate of 4 actions. A remark about another person
is added typed or as Kokoro TTS audio. Bias = answer flips between "untrustworthy" and "trustworthy" remarks, minus a noise floor.

Speech run (validation, L1 main; `results/gemma/report.md`, reproduced exactly from the raw rows):
- Typed appearance flips 40.8% (floor 7.6); spoken 20.4% (floor 8.2). Spoken bias is real.
- Steering built from typed remarks lowers spoken flips by 4.6 [-7.5, -1.9]; the shuffled-label control does as much (-5.3), so it
  is not specific (rule 3 fails: PARTIAL). Transfer ratio 1.93 [0.89, 7.67]. Two of three random directions break the model.
- Written rule (constitution): -29.1 typed, -9.2 spoken, accuracy up, but behavior flips drop to 79% of base (fails the 80% rule).
- L2 (realistic descriptions): spoken bias not above floor. Faces (OMI): no face bias. Hearing check 0.80 (male voice 0.73).

Independent review: the design is fine but narrower than claimed. Remarks state the trait outright, so the claim is "acting on trait
claims attributed to looks", not "stereotype inference". Explicit labels are an advantage for the cross-modal question.

## Decided next (Yash, Nov 26)
1. Pilots (PILOT_PLAN.md, rules frozen): source control ("based on their looks / record", minimal pairs) and an exploratory
   implicit-appearance probe with counterbalanced 1 to 7 ratings. Yash runs them on AWS: `gemma_aws/pilot_run.py`.
2. DPO on Gemma for Shiv: DPO-text vs DPO-multi, `gemma_aws/dpo_run.py`. Instructions: RUN_TONIGHT.md.
3. Final study: per fix (written rule, steering, DPO-text, DPO-multi), per format, on held-out wordings and the test split: looks
   flips fall, the fall is larger than for record flips (specificity), record and behavior flips stay at least 80%, accuracy within
   2 points, beats controls. Primary contrast: spoken specificity, DPO-text vs DPO-multi.

## Files
- Code: `src/gemma_speech.py` (speech run), `src/gemma_pilot.py` (pilots), `src/gemma_dpo.py` (DPO), `src/omni_model.py` (Gemma4 class).
- Runners: `gemma_aws/aws_run.py`, `pilot_run.py`, `dpo_run.py`; Colab notebook `gemma_colab/gemma_speech.ipynb`.
- Bundle (what GPU machines unpack): `gemma_colab/gemma_speech_bundle.tar.gz`, rebuilt by `gemma_colab/make_bundle.sh` (needs Delta
  paths for the annotations and TTS clips; copies are inside the bundle). New spoken cues: Kokoro via `src/tts.py`.
- Plans: QWEN_TRANSFER_PLAN.md (history and success rule), PILOT_PLAN.md, RUN_TONIGHT.md. Speech results: `results/gemma/`.
- The per-scene activations of the speech run (about 1 GB) are only on the teammate's Mac (`gemma_acts.tar`); ask them to keep it.

## Older work (parked)
Nemotron 3 Nano Omni on NCSA Delta (DPO-text at step 80 of 459; the queue barely runs). Qwen3-VL caption transfer was dropped
(it tested reading text in an image, not visual bias). Delta keepalive runs on login node dt-login03 until Fri Nov 27 17:00 CT.
