# sourced by every job
W=/work/hdd/bjuv/ylothe/idea3_final
export HF_HOME=/work/hdd/bjuv/ylothe/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export P777=$W DATA_DIR=$W/data/egonormia REF_DIR=$W/data/ref CACHE_DIR=$W/cache
export HF_MODULES_CACHE=/tmp/i3env_ylothe/hf_modules TRITON_CACHE_DIR=/tmp/i3env_ylothe/triton PYTHONUNBUFFERED=1
# environment lives on node-local disk (the shared filesystem is too slow for many small files)
if [ ! -f /tmp/i3env_ylothe/.ok ]; then mkdir -p /tmp && tar xf $W/i3env.tar -C /tmp && touch /tmp/i3env_ylothe/.ok; fi
export PATH=/tmp/i3env_ylothe/env/bin:$PATH
