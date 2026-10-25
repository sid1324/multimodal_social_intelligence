"""Verify pinned snapshot file sizes before allowing offline GPU jobs."""
import json
import hashlib
from pathlib import Path
import requests
from social_rl.common import STORE, config, write_json

cfg = config()
path = STORE / "models/Qwen3.5-9B"
index = json.loads((path / "model.safetensors.index.json").read_text())
weights = sorted(set(index['weight_map'].values()))
url = f"https://huggingface.co/api/models/{cfg['model_id']}/tree/{cfg['model_revision']}"
r = requests.get(url, timeout=60)
r.raise_for_status()
remote = {item['path']: item for item in r.json() if item['type'] == 'file'}
for name in weights:
    assert (path/name).stat().st_size == remote[name]['size'], name
    expected = remote[name].get('lfs', {}).get('oid')
    if expected:
        digest = hashlib.sha256()
        with (path/name).open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, name
required = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json', 'chat_template.jinja']
for name in required:
    assert (path/name).is_file(), name
write_json(path/'READY', {'model': cfg['model_id'], 'revision': cfg['model_revision'],
                        'weight_files': {n: {'bytes': remote[n]['size'], 'lfs': remote[n].get('lfs')} for n in weights},
                        'verification': 'Pinned HF snapshot, remote file sizes, and LFS SHA256 digests'})
print('MODEL_READY', path)
