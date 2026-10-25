# README for Yash (read this, Claude Code reads CLAUDE.md)

## What's in this packet
| Path | What it is |
|---|---|
| `CLAUDE.md` | Standing instructions Claude Code loads automatically: how to work with you, integrity rules, repo layout |
| `START_PROMPT.md` | Paste its middle section as your first message |
| `docs/01_PROJECT_BRIEF.md` | The idea, the normative rule, RQs, method, notation |
| `docs/02_CURRENT_STATE.md` | History, feedback, A2 findings, all 15 approved decisions |
| `docs/03_MIDTERM_PLAN.md` | Step-by-step experiments, gates, the pre-registered selection rule, the time-priority list |
| `docs/04_PSC_SETUP.md` | Bridges-2 commands (verify partition names with `sinfo`) |
| `docs/05_REPORT_REQUIREMENTS.md` | What the report needs from the experiments |
| `config/cues.yaml` | Every cue sentence (appearance / behavior / state, train vs. held-out) |
| `config/constitution.txt` | The constitution system prompt |
| `reference/course_docs/` | Midterm instructions, proposal, slides, A2 report, Oct 3 overview (the DPO parts are superseded for the midterm) |
| `reference/premise_analysis/` | Your A2 code and results (no images, **no .env**) |

## Things only you can add
1. **`.env`** (gateway URL and key). I deliberately left it out. Copy it to PSC yourself (command in `docs/04_PSC_SETUP.md`). Never commit it.
2. ~~PSC charge ID~~: done, `cis260338p` is filled in throughout.
3. **GPU availability:** whether A100/H100 exist on your allocation (V100 works either way).
4. **Optional:** the original-instructions PDFs for A2 and the presentations, and your Harvard paper PDF if you want Claude Code to cite its exact wording. Not needed for running experiments.
5. **The 40+20 face spot-check labels** during Step 2a. About 15 minutes of your time, and it counts as student-led work in the disclosure.

## Where to put it
Drag the zip and `.env` onto `idea3_face_bias/` in the team repo (VS Code Remote-SSH), unzip there, rename the folder to `midterm_steering`, move `.env` inside it, `chmod 600` it, and add `.env` and `midterm_steering/reference/` to `idea3_face_bias/.gitignore`. Open `idea3_face_bias/midterm_steering` in VS Code and start Claude Code there. Allocation `cis260338p`, `P777=/ocean/projects/cis260338p/ylothe` (already in your `~/.bashrc`).

## Checkpoints where you must look
1. Smoke test numbers (Step 0)
2. **Gate verdict (Step 1):** go or no-go
3. Face set counts, plus your spot-check (Step 2)
4. **Freeze (Step 3):** approve before the test run
5. SUMMARY.md and failure_cases.md: bring them back to the chat for writing
