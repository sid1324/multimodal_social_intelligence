# Final-study runs (plan B, Nov 27, 2026)

Every job is one command, resumable (run it again after any stop), and never touches the test split.
Branch `idea3-final`, folder `idea3_face_bias/final_omni/`.

## Setup on any Linux GPU machine (AWS, Colab terminal, lab server)
    git clone -b idea3-final --depth 1 https://github.com/sid1324/multimodal_social_intelligence.git   # code and bundle
    cd multimodal_social_intelligence/idea3_face_bias/final_omni
    cp gemma_colab/gemma_speech_bundle.tar.gz gemma_aws/       # the runner looks for the bundle next to itself
    cd gemma_aws && tmux new -s g                               # keeps running if the connection drops

GPU memory: Gemma needs 40 GB or more (A100 40/80, L40S, G4); Qwen3-Omni needs 80 GB or more (AWS g7e.2xlarge, Colab G4).
First run on a machine: about 15 to 30 minutes of setup (packages, images, model download).

## The three jobs
| Who | Command (inside gemma_aws/) | GPU | Time |
|---|---|---|---|
| Atharv | `python3 main_run.py --model gemma --job fixeval --dpo-zip <path to dpo_results_1127_1426.zip>` | any GPU with 40 GB or more; the zip from his DPO run supplies the adapters (not needed if the machine still has ~/idea3_gemma_speech/gs/final_omni/results/gemma_dpo) | 3 to 4 h |
| Shiv | `python3 main_run.py --model qwen --job dpo` | 96 GB (g7e / G4) | 8 to 12 h |
| Yash | `python3 main_run.py --model qwen --job steering` | 96 GB (the Qwen pilot instance) | 8 to 10 h |

What each does:
- Gemma fixeval: written rule, steering (+ weather and shuffled controls), DPO-text and DPO-multi on "based on their looks / record"
  claims (held-out wording), behavior sentences, noise floor and accuracy, typed and spoken, 196 validation scenes; checks the frozen rules.
- Qwen dpo: base answers on the training scenes, pairs, DPO-text (typed pairs) and DPO-multi (typed + spoken), 3 epochs each,
  evaluation of every epoch on validation (L1, typed and spoken), report with the epoch chosen by the midterm rule.
- Qwen steering: model check, hearing check, 300 training scenes (L1, L2), directions from typed remarks, settings sweep on typed
  validation remarks, the chosen setting applied to speech with controls and the written rule, faces, report.

Each run ends with `RESULTS ZIP: ~/idea3_gemma_speech/<model>_<job>_results_*.zip`. Send that zip to Yash (or put it in gemma_aws/ on Delta).

## Afterwards
Qwen fixeval needs the Qwen steering and DPO results on one machine; Claude will combine them. Then the freeze (Yash approves) and the
test-split run. The fixeval log prints which DPO adapters it found; if it says NONE, rerun with --dpo-zip.

## Please do not
Change settings, cues or rules without telling Yash. Do not run anything on the test split.
