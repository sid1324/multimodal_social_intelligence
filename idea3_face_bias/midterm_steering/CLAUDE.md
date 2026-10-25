# CLAUDE.md: Idea 3 (Physiognomic Bias Steering), CMU 11-777 Midterm

You are helping Yash Lothe (ylothe) build and run the midterm baseline for his solo research idea (Idea 3) in the Team 23 / "Team Socials" 11-777 Multimodal Machine Learning course project at CMU.

**Read these first, in order:**
1. `docs/01_PROJECT_BRIEF.md`: what the idea is, the research questions and the normative rule
2. `docs/02_CURRENT_STATE.md`: the history, what Milestone 2 (A2) found, and every decision already made
3. `docs/03_MIDTERM_PLAN.md`: the exact experiments, specs, gates and acceptance checks (**this is the work**)
4. `docs/04_PSC_SETUP.md`: the compute environment (PSC Bridges-2)
5. `docs/05_REPORT_REQUIREMENTS.md`: what the midterm report needs from these experiments
6. `config/cues.yaml` and `config/constitution.txt`: the exact cue sentences and constitution text


## How to work with Yash
- **Work in chunks.** For each step, give a short plan, run it, show the results in simple words, then propose the next step. Yash wants to understand every consequential step.
- **Explain things simply.** Short sentences, no jargon without a one-line definition.
- **Every terminal command you give Yash comes with a one-line explanation of what it does.** No exceptions.
- **Never use em dashes** in any prose you write: reports, summaries, docs, messages or comments. Use commas, colons, parentheses or separate sentences.
- **Ask before:** any paid or gateway (frontier-model) API calls beyond what a step's budget says, regenerating or editing face images, changing the pre-registered selection rule or test conditions, or anything that touches the test split before the validation-chosen settings are frozen.
- **Proceed without asking for:** reading files, writing code, unit tests, small smoke tests on the train split, submitting the planned Slurm jobs.

## Scientific integrity rules (non-negotiable)
- **The test split is touched once,** after the steering settings are frozen on validation. Log the freeze, with a timestamp, in `results/FREEZE.md`.
- **Never invent data,** human labels or ratings. Human spot-checks are done by Yash. Build him a labeling page; never fill it in yourself.
- **Report failures honestly.** A failed control is a result, not something to hide. The midterm rubric weights failure analysis most.
- **No causal language beyond the design.** The face arm measures face *sensitivity* and its *alignment* with the text trust axis. It does not isolate facial morphology, because real faces differ in expression too. Say so.
- **Keep a running `AI_USAGE_LOG.md`** (what you wrote, ran, decided or suggested, with dates). The course requires a detailed AI-usage disclosure, and this log feeds it.

## PSC rules (details in docs/04_PSC_SETUP.md)
- **Wrong-allocation risk:** Yash also has the 11-868 allocation `cis260267p`. Never use it. Never use `$PROJECT`. Use `$P777` and `-A cis260338p` everywhere.
- Use a uv venv in `$P777`, `module load cuda/12.4.0`, and caches in `$P777`.
- Use v100-32 with `--exclude=v005` (a known broken node). Avoid v100-16, because 8B fp16 won't fit.
- Run big downloads with `nohup` (login-node internet can crawl). Keep jobs short and sharded. Check `squeue --start` for congestion.
- Keep the code path-agnostic (env vars for data, cache and output) so a switch to the Colab A100 fallback takes minutes.

## Repository and secrets
- The team repo is `github.com/sid1324/multimodal_social_intelligence`. All midterm Idea 3 code lives in `idea3_face_bias/midterm_steering/` (this folder). The sibling folders in `idea3_face_bias/` (`code/`, `earlier_pipeline/`, `figures/`, `results/`) are the A2 work: read-only reference, never modify. Commit small and often with clear messages, because GitHub history is used to assess contribution.
- **Never commit or print** `.env` (it holds `LITELLM_BASE_URL` and `LITELLM_API_KEY` for the CMU AI gateway). Never commit data, images, model weights or activation caches. Add them to `.gitignore`.
- The Milestone 2 code (`premise_analysis/`) is reference material. Reuse its logic (prompt format, letter parsing, gateway client, lexicon, bootstrap). Do not modify its archived results.

## Suggested layout
```
idea3_face_bias/midterm_steering/
  CLAUDE.md, docs/, config/            # this packet
  src/
    data.py          # load annotations, build prompts, splits, option permutations
    model.py         # load Qwen3-VL, option-letter log-probs, hooks for direction ablation
    faces.py         # face candidate selection, blur and background-blur conditions
    gateway.py       # frontier-model calls (face verification, trust/expression ratings)
    extract.py       # activation collection and direction estimation
    evaluate.py      # run condition x method grids, write JSONL
    analyze.py       # metrics, bootstrap CIs, figures, failure tables
  slurm/             # sbatch scripts
  results/           # small JSON/CSV/figures (commit these); big caches go in $P777 (the 11-777 project folder), not git
  AI_USAGE_LOG.md
```
