set -x
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python eval_combined.py base rgb results/base_rgb.json --batch 8
python eval_combined.py base combined results/base_combined.json --batch 4
python eval_combined.py adapters/spatial combined results/spatial_s0_combined.json --batch 4
python eval_combined.py adapters/spatial_s2 combined results/spatial_s2_combined.json --batch 4
python eval_combined.py adapters/spatial rgb results/spatial_s0_rgb.json --batch 8
python eval_combined.py adapters/spatial_s2 rgb results/spatial_s2_rgb.json --batch 8
echo ALL_DONE
