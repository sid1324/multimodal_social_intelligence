"""Pinned Qwen3.8-27B download; separate from the established 9B environment."""
import json,os
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path('/ocean/projects/cis260338p/apatawar/social_norm_rl/models/Qwen3.8-27B')
rev='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
os.environ.pop('HF_HUB_OFFLINE',None)
snapshot_download('Qwen/Qwen3.8-27B',revision=rev,local_dir=root,
 allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','README.md'],max_workers=2)
(root/'READY').write_text(json.dumps({'repository':'Qwen/Qwen3.8-27B','revision':rev}))
