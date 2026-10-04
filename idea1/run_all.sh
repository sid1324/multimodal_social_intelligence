#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${ANALYSIS_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6
export MPLCONFIGDIR="${TMPDIR:-/tmp}/analysis1-matplotlib"
export ANALYSIS_MODELS="$(pwd)/models"
mkdir -p outputs/logs outputs/tables outputs/figures qualitative
"$PYTHON" scripts/01_reaudit.py > outputs/logs/01_reaudit.log 2>&1
"$PYTHON" scripts/02_probes.py > outputs/logs/02_probes.log 2>&1
"$PYTHON" scripts/03_external_audits.py > outputs/logs/03_external_audits.log 2>&1
"$PYTHON" scripts/05_embeddings.py text > outputs/logs/05_text.log 2>&1
"$PYTHON" scripts/05_embeddings.py visual > outputs/logs/05_visual.log 2>&1
"$PYTHON" scripts/06_sensitivity.py > outputs/logs/06_sensitivity.log 2>&1
"$PYTHON" scripts/select_cases.py > outputs/logs/select_cases.log 2>&1
"$PYTHON" scripts/07_figures.py > outputs/logs/07_figures.log 2>&1
"$PYTHON" scripts/08_matched_semantics.py > outputs/logs/08_matched_semantics.log 2>&1
"$PYTHON" scripts/09_human_packet.py > outputs/logs/09_human_packet.log 2>&1
"$PYTHON" scripts/11_human_results.py > outputs/logs/11_human_results.log 2>&1
"$PYTHON" scripts/12_human_validate.py > outputs/logs/12_human_validate.log 2>&1
"$PYTHON" scripts/10_validate.py > outputs/logs/10_validate.log 2>&1
printf 'Analysis rerun complete. See outputs/tables and qa.\n'
