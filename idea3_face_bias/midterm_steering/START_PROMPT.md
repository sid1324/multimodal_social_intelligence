# Paste this as your first message to Claude Code

---

Hi. I'm Yash. This folder (`idea3_face_bias/midterm_steering/`) is the full context packet for my 11-777 midterm work (Idea 3: physiognomic-bias steering on EgoNormia with Qwen3-VL-8B). The PSC allocation is `cis260338p` with `P777=/ocean/projects/cis260338p/ylothe`. Never use `cis260267p` (my 11-868 allocation) or `$PROJECT`.

**Before anything else (the model download is the longest step):**
1. Set up the uv venv at `$P777/.venv` (caches in `$P777/.cache`), with `module load cuda/12.4.0`, and install `huggingface_hub`.
2. Start a background `nohup` download of `Qwen/Qwen3-VL-8B-Instruct` with `HF_HOME=/ocean/projects/cis260338p/shared/hf_cache`, logging to `$P777/model_download.log`. Verify the exact model id exists first.
3. Show me the log after about 30 seconds so I can confirm it's running.
4. Check that `.env` is in this folder and that `.gitignore` covers `.env` and `reference/`.

**Then:**
1. Read `CLAUDE.md`, then `docs/01` through `docs/05` and `config/` in order. Skim `reference/course_docs/MidtermAssignment-Instructions-Report.pdf`, the A2 report (`reference/course_docs/A2_Milestone2_Report_Team23.pdf`, Section 4 and the Idea 3 paragraph of Section 7), and `reference/premise_analysis/pilot.py` (prompt format and gateway client to reuse).
2. Run `date` and tell me where we stand against the timeline in `docs/03_MIDTERM_PLAN.md`.
3. In plain words, give me: (a) your understanding of the goal in 5 bullets, (b) anything in the packet that looks wrong, risky or ambiguous, (c) your exact plan for the rest of Step 0, with any commands I need to run myself (each with a one-line explanation).

Don't write experiment code until I've confirmed (a) to (c). Deadline: Idea 3 results go to my team Sunday 12:00 noon ET, so keep it brief.

---

**After that, typical follow-ups:**
- "Go ahead with Step 0." Then review the smoke test numbers.
- "Show me the gate results in simple words." This is the go/no-go before steering.
- "Show me the val sweep and the configs you'd freeze." Approve before the test run.
- "Write SUMMARY.md and failure_cases.md." Then bring those files back to the chat Claude to write the report.
