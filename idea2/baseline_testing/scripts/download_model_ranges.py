"""Resumable HTTP-range fallback for the pinned official model weight shards.

Use only after stopping scripts/download_model.py. This preserves its existing
Xet partials and verifies every assembled shard against the official LFS SHA.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

import requests

from social_rl.common import STORE, config


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_range(url, start, end):
    last = None
    for attempt in range(6):
        try:
            response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(20, 45))
            response.raise_for_status()
            expected_header = f'bytes {start}-{end}/'
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(expected_header):
                raise IOError('Model host ignored requested byte range')
            if len(response.content) != end + 1 - start:
                raise IOError('Short model byte-range response')
            return response.content
        except (requests.RequestException, OSError) as error:
            last = error
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f'Cannot fetch model bytes {start}-{end}: {last}')


def matching_prefix(partial, url):
    size = partial.stat().st_size
    if not size:
        return False
    with partial.open('rb') as f:
        for start in (0, max(0, size - 1024)):
            f.seek(start)
            width = min(1024, size - start)
            if f.read(width) != get_range(url, start, start + width - 1):
                return False
    return True


def main():
    cfg = config()
    directory = STORE / 'models/Qwen3.5-9B'
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    api = f"https://huggingface.co/api/models/{cfg['model_id']}/tree/{cfg['model_revision']}"
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    remote = {item['path']: item for item in response.json() if item['type'] == 'file'}
    for name in names:
        entry = remote[name]
        total = entry['size']
        oid = entry['lfs']['oid']
        final = directory / name
        if final.exists() and final.stat().st_size == total and digest(final) == oid:
            print('VERIFIED_EXISTING', name, flush=True)
            continue
        url = f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['model_revision']}/{name}"
        part = final.with_suffix(final.suffix + '.http.part')
        if not part.exists():
            shard_number = name.split('-')[1]
            saved = STORE / 'cache/model_partials' / f'shard-{shard_number}.prefix'
            candidates = [saved] + list((directory / '.cache/huggingface/download').glob(f'*.{oid}.incomplete'))
            for candidate in candidates:
                if candidate.exists() and matching_prefix(candidate, url):
                    shutil.copyfile(candidate, part)
                    print('COPIED_VALID_PREFIX', name, part.stat().st_size, flush=True)
                    break
        for restart in range(2):
            if part.exists() and part.stat().st_size > total:
                part.unlink()
            offset = part.stat().st_size if part.exists() else 0
            while offset < total:
                end = min(offset + 4 * 1024 * 1024, total) - 1
                block = get_range(url, offset, end)
                with part.open('ab') as f:
                    f.write(block)
                offset = end + 1
                if offset % (256 * 1024 * 1024) < len(block) or offset == total:
                    print('SHARD_PROGRESS', name, offset, total, flush=True)
            if digest(part) == oid:
                part.replace(final)
                print('SHARD_VERIFIED', name, total, flush=True)
                break
            print('SHA_MISMATCH_RESTART', name, flush=True)
            part.unlink()
        else:
            raise RuntimeError(f'SHA256 mismatch for {name} after full retry')


if __name__ == '__main__':
    main()
