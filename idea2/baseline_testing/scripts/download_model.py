import os
from social_rl.common import STORE, config
os.environ['HF_HOME'] = str(STORE / 'cache/huggingface')
os.environ.pop('HF_HUB_OFFLINE', None)
from huggingface_hub import snapshot_download

cfg = config()
print(snapshot_download(cfg['model_id'], revision=cfg['model_revision'],
    local_dir=STORE/'models/Qwen3.5-9B',
    allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md'], max_workers=2), flush=True)
