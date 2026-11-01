"""Extract eligible PsycheMME frames and make grouped source-only SFT splits."""
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from social_rl.common import STORE, write_json, write_jsonl

REV = '3b80a47473802b160b54e760d5dca070efd96f4c'


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def group(ep):
    source, vid = ep['source_dataset'], ep['source_video_id']
    if source == 'EgoCom':
        m = re.match(r'vid_\d+__day_(\d+)__con_(\d+)__person_\d+', vid)
        if m:
            return f'{source}:day_{m[1]}:con_{m[2]}'
    return f'{source}:{vid}'


def main():
    root = STORE / 'raw/psychemme' / REV
    out = STORE / 'data/psychemme'
    audit = out / 'audit'
    ids = set((audit / 'eligible_sft_ids.txt').read_text().splitlines())
    eps = {x['episode_id']: x for x in read(root / 'metadata/episodes.jsonl')}
    qs = {x['id']: x for x in read(root / 'metadata/questions.jsonl')}
    rows = []
    needed = set()
    for x in read(root / 'sft/train.jsonl'):
        if x['id'] not in ids or qs[x['id']]['task_type'] != 'single_choice':
            continue
        ep = eps[qs[x['id']]['episode_id']]
        frames = [x['images'][i] for i in (0, 4, 9, 13, 18, 22, 27, 31)]
        needed.update(frames)
        g = group(ep)
        v = int(hashlib.sha256(('psychemme-sft-v1:' + g).encode()).hexdigest()[:12], 16) / 16**12
        split = 'val' if v < .1 else 'train'
        target = json.loads(x['messages'][-1]['content'])
        observation = next(str(v) for k, v in target.items() if k != 'answer' and v)
        user = x['messages'][0]['content']
        user = user[user.index('Timestamped transcript'):user.index('Slot meanings:')].strip()
        if re.search(r'\b(?:option|choice|answer)\s+[A-J]\b', user, re.I):
            continue  # Option-letter references would become wrong after permutation.
        options = re.findall(r'^([A-Z])\. (.*)$', user, re.M)
        answer = target['answer']
        if not options or len(answer) != 1 or answer not in {a for a, _ in options}:
            raise ValueError(f'Unexpected answer/options: {x["id"]}')
        rows.append({'id':x['id'],'split':split,'group':g,'source':ep['source_dataset'],
                     'category':qs[x['id']]['category'],'prompt':user,
                     'answer':answer,'observation':observation,'frames':frames})
    archive = root / 'media/frames.tar'
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):
            h.update(part)
    if h.hexdigest() != 'beaa87b53b33acc6788a18f43250b39f6dcfc20e963115c9637dd75440de2674':
        raise ValueError('Frame archive checksum mismatch')
    extracted = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name.removeprefix('./')
            if name not in needed:
                continue
            if not member.isfile() or not name.startswith('media/frames/') or '..' in Path(name).parts:
                raise ValueError(f'Unsafe archive member {name}')
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as dest:
                while chunk := source.read(1024*1024):
                    dest.write(chunk)
            extracted.add(name)
    if extracted != needed:
        raise ValueError(f'Missing frames {len(needed-extracted)}')
    for x in rows:
        x['frames'] = [str(out / p) for p in x['frames']]
    for split in ('train','val'):
        write_jsonl(out / f'{split}.jsonl', [x for x in rows if x['split']==split])
    write_json(out / 'split_audit.json', {'revision':REV,'frame_tar_sha256':h.hexdigest(),
        'counts':dict(Counter(x['split'] for x in rows)),
        'categories':{s:dict(Counter(x['category'] for x in rows if x['split']==s)) for s in ('train','val')},
        'source_groups':{s:len({x['group'] for x in rows if x['split']==s}) for s in ('train','val')},
        'excluded_open_questions':sum(qs[x]['task_type']=='open' for x in ids),
        'frames_extracted':len(extracted)})
    print((out / 'split_audit.json').read_text(),flush=True)

if __name__=='__main__':
    main()
