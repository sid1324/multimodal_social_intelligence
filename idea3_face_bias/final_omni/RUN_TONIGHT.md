# Running the pilots and the Gemma DPO (Nov 26, 2026)

Both need one GPU with at least 40 GB (AWS g6e.xlarge, Colab G4/A100, any lab server). Both are resumable: if anything stops,
run the same command again. Neither touches the test split. Branch: `idea3-final`, folder `idea3_face_bias/final_omni/`.

## Get the code onto the GPU machine
    git clone -b idea3-final --depth 1 https://github.com/sid1324/multimodal_social_intelligence.git   # code and bundle
    cd multimodal_social_intelligence/idea3_face_bias/final_omni
    cp gemma_colab/gemma_speech_bundle.tar.gz gemma_aws/            # the runners look for the bundle next to themselves
    cd gemma_aws && tmux new -s g                                    # keeps running if the connection drops

The first run on a machine installs packages and downloads Gemma (about 15 to 25 minutes). Later runs reuse them.

## 1. Pilots (Yash): about 1 hour after setup
    python3 pilot_run.py

What it runs (PILOT_PLAN.md, rules frozen before any result): a hearing check for the source words, then 300 training scenes
typed and spoken with the source cues ("based on their looks / record", no source) and the implicit appearance cues, then 1 to 7
trust ratings (asked in both scale directions). It prints the report with the frozen rules (Pilot 1 GO or NO GO; Pilot 2 include
or not) and writes `~/idea3_gemma_speech/pilot_results_*.zip`. Send that zip.

## 2. DPO on Gemma (Shiv): about 5 to 8 hours after setup
    python3 dpo_run.py

What it runs (`src/gemma_dpo.py`, settings `config/dpo.yaml`, same as the Nemotron plan): base answers on the remaining training
scenes (the 300 gate scenes come from the finished speech run, included in the bundle), pair building, DPO-text (typed pairs),
DPO-multi (the same pairs typed and spoken), evaluation of every epoch on validation (L1, typed and spoken), report with the epoch
chosen by the midterm rule. Output: `~/idea3_gemma_speech/dpo_results_*.zip` (report, logs, answer rows, LoRA adapters). Send that
zip. Expected size: about 4,000 to 5,000 pairs; DPO-text about 1 to 2 h, DPO-multi about 2 to 4 h, evaluation about 1 h.

## How this was tested
- Pilots: every stage end to end through `pilot_run.py` in a sandbox with a stand-in model; real Gemma 4 on CPU scored a spoken
  pilot cue cleanly (letter mass 0.9999) and the rating prompt (digits 1 to 7 are single tokens).
- DPO: every stage end to end through `dpo_run.py` with a tiny randomly initialized Gemma 4 (same architecture and processor):
  pairs, LoRA training with gradient checkpointing, checkpoints, adapter saving and loading, evaluation, report.
- Not tested on a GPU yet: speed and memory of 12B DPO training. If it runs out of memory, tell Yash (it should fit: about 24 GB
  of weights, LoRA only, one example at a time with gradient checkpointing).

## Please do not
Change settings, cues or rules without telling Yash: the decision rules were fixed before results. Do not run anything on the test
split: that needs Yash's approval of a freeze first.
