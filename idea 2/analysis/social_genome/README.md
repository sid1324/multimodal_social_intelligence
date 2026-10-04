# Active Social Genome analysis

See the [package README](../../README.md) for setup, completed results, excluded
data and reproduction instructions. Run commands from the `idea 2/` directory.

```bash
python -m analysis.social_genome --help
python -m analysis.social_genome run
```

`config.json` selects the completed single-annotator audiovisual study.
`config_legacy_silent.json` describes the earlier silent-video packets.
The source IDs are frozen in `study_ids.csv`; changes explicitly confirmed by the
human are in `annotation_corrections.json`. Source media and human responses are
not distributed in this repository.
