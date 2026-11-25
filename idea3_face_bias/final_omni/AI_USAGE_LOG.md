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
| Nov 24 01:00 to 08:10 | Claude | Implemented and ran overnight steps 1 to 3 (through G2 text), wrote and debugged the fast MoE dispatch check, the driver and keepalive, partial speech/image. Decisions made by Claude: newline answer prefix, batching off, A40/H200 use, extending the submission window to 07:30 CT. |
| Nov 24 13:50 ET | Yash | Approved: L2 main with L1 reported beside it and DPO on both; G1 setup (newline prefix, one at a time, mixed GPUs, batching off); remark framing and the constitution sentence; 200 SU total with a pause at 150; chained interactive jobs; staged freeze (steering first, test results of base/constitution/steering unread until DPO is frozen); VideoNorms deferred. Test runs still need Yash's review of each freeze. |
| Nov 24 13:50 ET | Claude | Set DPO micro_batch to 1 (right-padded micro-batches use the path that failed G1(b)); 57-min jobs with two copies on 4x A40; exploratory L1 speech/image; order: sweep first, DPO training rows alongside (base scoring only). |
| Nov 25 16:20 ET | Yash | Decision: run the DPO study on Qwen2.5-Omni-7B on Colab (A100 40 GB), because Delta queue time blocked DPO on Nemotron. Overrides D2b (7B was for debugging only). Nemotron stays the 30B model for bias, image use and directions. |
| Nov 25 evening | Yash | Approved the new plan: test whether midterm steering on Qwen3-VL 8B transfers from text to image remarks on Colab (speech later with an omni model); if successful, delegate bigger models to Atharv and Shiv. Delta jobs keep running. Success rule and random-direction control proposed by Claude, pending Yash's review before any steered image run. |
