"""Recreate the pinned annotation/metadata inputs without downloading videos."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import requests
os.environ.pop('HF_HUB_OFFLINE', None)
from huggingface_hub import hf_hub_download

from social_rl.common import ROOT, STORE, config


EXPECTED = {
    'r3.parquet': '4a15b11efa4dbf2d6b0539bfda54a413a4c3877c42c4f526efb8f88e7c2d05fc',
    'r3_durations.csv': 'd5fbdc6d532863ab84890d50953e1a4b4f1c63c202318eeb568be29bf2b2fb7e',
    'ego.parquet': '0dce88b77f3c0f911b015466d0efac211d5d71f71f956294de361c7b5aa97cb7',
    'condensed_clips.csv': '02958a978af5075871a557b28f7aa9f3fca56f4cdba64178f25d5a4539f0f140',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_copy(source, target):
    if target.exists():
        if sha256(target) != EXPECTED[target.name]:
            raise RuntimeError(f'Existing raw file checksum differs: {target}')
        print('VERIFIED_EXISTING', target, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if sha256(source) != EXPECTED[target.name]:
        raise RuntimeError(f'Pinned source checksum differs: {source}')
    shutil.copyfile(source, target)
    print('COPIED', target, flush=True)


def main():
    cfg = config()
    raw = STORE / 'data/raw'
    cache = STORE / 'cache/huggingface/hub'
    for repo, revision, remote, local in (
        ('nlxpku/R3-FDT', cfg['r3_revision'], 'qas.parquet', 'r3.parquet'),
        ('nlxpku/R3-FDT', cfg['r3_revision'], 'videos_and_durations.csv', 'r3_durations.csv'),
        ('open-social-world/EgoNormia', cfg['ego_revision'], 'train-norm-updated.parquet', 'ego.parquet'),
    ):
        target = raw / local
        if target.exists() and sha256(target) == EXPECTED[local]:
            print('VERIFIED_EXISTING', target, flush=True)
            continue
        source = hf_hub_download(repo, remote, repo_type='dataset', revision=revision, cache_dir=cache)
        checked_copy(source, target)

    target = raw / 'condensed_clips.csv'
    if target.exists() and sha256(target) == EXPECTED[target.name]:
        print('VERIFIED_EXISTING', target, flush=True)
    else:
        url = ('https://raw.githubusercontent.com/m-bain/CondensedMovies/'
               f"{cfg['condensedmovies_commit']}/data/metadata/clips.csv")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp')
        temp.write_bytes(response.content)
        if sha256(temp) != EXPECTED[target.name]:
            temp.unlink()
            raise RuntimeError('Pinned CondensedMovies metadata checksum differs')
        temp.replace(target)
        print('COPIED', target, flush=True)

    repo = ROOT / 'vendor/EgoNormia'
    commit = cfg['ego_repository_commit']
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/Open-Social-World/EgoNormia.git', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'fetch', '--depth=1', 'origin', commit], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', commit], check=True)
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != commit:
        raise RuntimeError(f'EgoNormia repository at {head}, expected {commit}')
    print('VERIFIED_EGONORMIA_REPO', commit, flush=True)


if __name__ == '__main__':
    main()
