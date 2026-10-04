# Idea 2 — Evidence-grounded reward feasibility

CMU 11-777, Team 23. Lead and sole human annotator: Atharv Patawar.

This is the completed Assignment 2 Social Genome feasibility pilot for Idea 2.
It contains analysis code, tests, frozen sample IDs, derived tables and figures.
No VLM training or GRPO was run, and this pilot did not use EgoNormia.

## Completed results

One person annotated 24 English clips from 24 source videos, using video, original
audio and English transcripts. The original export had one modality inconsistency;
the human-confirmed correction is recorded in
[annotation_corrections.json](analysis/social_genome/annotation_corrections.json).
The public Social Genome inputs have no gold answers or reference evidence.

| Human-reported measure | Items |
|---|---:|
| Clearly observable evidence | 24/24 |
| Temporal reasoning | 22/24 |
| Visual information used | 22/24 |
| Transcript used | 21/24 |
| Audio used | 7/24 |

The post-hoc human-description/input-transcript diagnostic retrieved the paired
transcript first for 11/24 descriptions and within the top five for 17/24. It measures
text association, not evidence correctness: the annotator had already seen the
transcripts, and visual observations need not appear in them. No inter-annotator
agreement, answer accuracy or validated evidence reward is claimed.

- [Final figures (PNG and PDF)](outputs/social_genome_audio/figures/)
- [Final tables](outputs/social_genome_audio/tables/)
- [Human summary](outputs/social_genome_audio/tables/human_statistics.json)
- [Language summary](outputs/social_genome_audio/tables/human_language_statistics.json)
- [Reported rates with Wilson intervals](outputs/social_genome_audio/tables/human_rates.csv)
- [Publication inventory and checksums](publication_manifest.json)

`outputs/social_genome/` contains earlier public-input diagnostics before the
single-annotator audiovisual study. The final results are in
`outputs/social_genome_audio/`.

## Run from this directory

All relative paths are relative to `idea 2/`, so quote its name:

```bash
cd 'idea 2'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis2.txt
python -m pytest -q
```

The active entry point is `python -m analysis.social_genome`; use `--help` for
commands. `analysis/idea2/` is the older schema-gated VideoNorms implementation
and shared utilities reused by the active pipeline. It has no empirical
VideoNorms results. The tests use synthetic temporary fixtures, never fake human
study responses.

## Data and full reproduction

The `docs/` directory, assignment instructions PDF, all source data and media,
individual annotation exports, download logs, and model/embedding caches are
intentionally excluded and covered by `.gitignore`. Tables that duplicated raw
question/transcript/response text are omitted or published as separate assignment
files containing only IDs and computed columns; see `publication_manifest.json`.
A document exclusion does not exclude the research figures' PDF exports.

To reproduce the exact completed human study, restore the excluded
`data/social_genome_audio/` study archive (including its frozen manifest, media
and corrected response CSV), then run:

```bash
python -m analysis.social_genome run
```

If importing the original human export, first run:

```bash
python -m analysis.social_genome import-responses --annotator annotator1 --file /path/to/original_export.csv
python -m analysis.social_genome apply-corrections --file analysis/social_genome/annotation_corrections.json
python -m analysis.social_genome run
```

The original human responses must be supplied separately; they cannot be
reconstructed or replaced by AI. With the encoder already cached, set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for an offline run.

To acquire the same source IDs and prepare a **new blank study**, install FFmpeg
and ffprobe on PATH, then:

```bash
python -m pip install yt-dlp==2026.8.19
python -m analysis.social_genome inspect
python -m analysis.social_genome fetch
python -m analysis.social_genome audio
python -m analysis.social_genome prepare
```

`prepare` refuses to overwrite an existing study. The downloader fetches only the
selected one-minute sections and enforces a raw-data budget. Exact replication
requires the archived study: live video availability, encodings and captions may
change even when IDs are fixed. The offline annotation page is generated at
`data/social_genome_audio/blinded/annotator1/index.html`. Export responses manually;
opening the page alone does not load saved annotations.

Dataset revision: `7646cde5c23e9acf7c2665d19f2f57bcbec0554c` from
[the official Social Genome repository](https://github.com/CMU-MultiComp-Lab/social-genome).
Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The code records actual package
versions and uses explicit token-window pooling instead of silent truncation.

## Attribution

Atharv directed and approved the work and supplied all 24 human annotations and
the explicit correction. Codex generated and executed substantial parts of the
pipeline, annotation interface, tests, figures and draft interpretation. The
full local AI disclosure and report documents were excluded from this upload at
the owner's request; this README does not imply the code was solely human-written.
