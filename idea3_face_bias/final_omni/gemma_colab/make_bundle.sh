#!/bin/bash
# Builds gemma_colab/gemma_speech_bundle.tar.gz: code, config, splits, annotations, and the Kokoro audio of every cue (no images, no model).
set -euo pipefail
cd "$(dirname "$0")/.."
W=/work/hdd/bjuv/ylothe/idea3_final
D=$(mktemp -d "${TMPDIR:-/tmp}/gsb.XXXX")
mkdir -p $D/final_omni/results $D/p777/data $D/p777/cache
cp -r src config $D/final_omni/ && rm -rf $D/final_omni/src/__pycache__
cp results/splits.json $D/final_omni/results/
cp -r $W/data/ref $D/p777/data/ref
cp -r $W/cache/tts $D/p777/cache/tts
tar czf gemma_colab/gemma_speech_bundle.tar.gz -C $D final_omni p777
echo "bundle: $(du -h gemma_colab/gemma_speech_bundle.tar.gz | cut -f1), $(tar tzf gemma_colab/gemma_speech_bundle.tar.gz | wc -l) entries, tmp dir $D"
