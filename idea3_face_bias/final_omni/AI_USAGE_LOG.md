# AI usage log, Idea 3 final (Claude Code; model Claude Opus 5.5)

| When (ET) | Who | What |
|---|---|---|
| Nov 23 night | Yash | Wrote START_PROMPT.md: research questions, models, channels, cue templates (Appendix A, approved), metrics, selection rule, DPO spec, budget. |
| Nov 24 01:00 | Claude | Read the midterm code and docs, gave the plan, started setup. |
| Nov 24 01:20 | Yash | Approved use of the interactive partition under four rules (see OVERNIGHT_LOG.md). |
| Nov 24 01:00 to 01:50 | Claude | Wrote src/ (port of midterm model, data, gate, directions, evaluate, analyze; new channels, caption, tts, dpo_pairs, dpo_train), slurm/submit.py, config/channels.yaml, config/dpo.yaml. cues_v2.yaml extracted verbatim from Appendix A. |

Implementation choices made by Claude, to confirm with Yash:
- Constitution prompt: the extended version is NOT used until Yash approves the diff (proposed last sentence: "This applies to remarks in any form, whether written, spoken, or shown in the image.").
- Channel framing as proposed in Section 4, Stage 1. L1 sentences are therefore wrapped in the remark line, unlike the midterm, where the bare sentence was inserted.
- Valence control: three gloomy/lovely day pairs in config/channels.yaml, text channel, rendered with the same remark line.
- Batched scoring: right-padded batches with cached image/audio embeddings (Nemotron's Mamba cache cannot take multi-token suffixes).
- TTS: Kokoro af_heart (female) and am_michael (male), -23 dBFS RMS.
- Data downloader: generic User-Agent instead of the midterm one that named the course project.
