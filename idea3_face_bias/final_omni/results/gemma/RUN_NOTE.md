# Gemma 4 12B speech-transfer run: Colab G4 (Nov 26, 2026)

- Hardware: Colab G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB), started with the Colab CLI (`colab new --gpu G4`).
- Code and settings: unchanged. Same `gemma_speech_bundle.tar.gz` as `gemma_colab/` on this branch (SHA-256 bac421db27fbef4e…), and the notebook's own `gs_colab.py` (cell 2). The stages are those of notebook cells 3, 5, 7 and 8: check, perc, g2, gate, directions, sweep, select, transfer, report, faces, report.
- Drive was not mounted. Results were synced here from the runtime as stages finished. The per-scene activation files (`acts/`, about 1 GB) are not in git. `directions_gemma.npz` is.
- Files in this folder update as the run continues. The run is complete when `report.md` and the `gemma_results_*.zip` are present.
- Hearing check (perception.json): speech 0.80 (female 0.84, male 0.73), typed 1.00. An aborted local Mac (MPS) attempt on the same 30 scenes gave speech 0.87, so treat this Colab run as the official one.
