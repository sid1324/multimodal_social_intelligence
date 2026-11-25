"""Data layer: annotations, cue sentences, prompts, per-item option permutations and video splits.

Everything here is deterministic. Seeds come from SHA-256 of strings (never Python's salted hash()).
Paths come from env vars so the code runs unchanged on PSC or Colab:
  DATA_DIR  (default $P777/data/egonormia)   images: model_previews/<id>.jpg
  REF_DIR   (default ./reference/premise_analysis)   A2 annotations and item table
"""
import csv, hashlib, json, os, random
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent.parent
P777 = os.environ.get('P777', '/ocean/projects/cis260338p/ylothe')
DATA = Path(os.environ.get('DATA_DIR', f'{P777}/data/egonormia'))
REF = Path(os.environ.get('REF_DIR', HERE / 'reference/premise_analysis'))
CONFIG = HERE / 'config'
LETTERS = 'ABCDE'
PRIMARY = {'Privacy', 'Coordination/Proactivity', 'Communication/Legibility', 'Cooperation'}
SPLIT_SEED = 'idea3-midterm-split-v1'
PERM_SEED = 'idea3-midterm-perm-v1'
SPLIT_FRACS = (0.50, 0.15, 0.35)  # train, val, test (by source video)

PORTRAIT_SENTENCE = 'The second image shows the face of the other person in the scene.'


def seeded_int(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def video_of(item_id):
    return item_id.split('_')[0]


def load_annotations():
    return json.loads((REF / 'data/official_annotations.json').read_text())


def load_item_table():
    return {r['id']: r for r in csv.DictReader(open(REF / 'results/text_arm_items.csv'))}


def gold_nonempty(v):
    return bool(v['behaviors'][v['correct']].strip())


def norms_of_gold(v):
    return sorted(v['taxonomy'].get(str(v['correct']), []))


def load_cues():
    """Flatten config/cues.yaml into {condition_name: sentence}. Template ids are explicit:
    app_low_tr0..2, app_high_tr0..2, app_low_ho0..1, app_high_ho0..1,
    beh_neg_tr0..1, beh_pos_tr0..1, beh_neg_ho0..1, beh_pos_ho0..1, state_neg0..1, state_pos0..1."""
    c = yaml.safe_load((CONFIG / 'cues.yaml').read_text())
    out = {'none': '', 'neutral': c['neutral'], 'neutral_paraphrase': c['neutral_paraphrase']}
    for split, tag in (('train', 'tr'), ('heldout', 'ho')):
        for side in ('low', 'high'):
            for i, s in enumerate(c['appearance'][split][side]):
                out[f'app_{side}_{tag}{i}'] = s
        for side in ('neg', 'pos'):
            for i, s in enumerate(c['behavior'][split][side]):
                out[f'beh_{side}_{tag}{i}'] = s
    for side in ('neg', 'pos'):
        for i, s in enumerate(c['state'][side]):
            out[f'state_{side}{i}'] = s
    return out


def constitution():
    return (CONFIG / 'constitution.txt').read_text().strip()


def options(v):
    """Non-empty options in original order: list of (original index, text)."""
    return [(i, b) for i, b in enumerate(v['behaviors']) if b.strip()]


def permuted_options(v):
    """One fixed random order per item (seeded by the item id), shared by every condition and method.
    Returns a list of (letter, original index, text)."""
    opts = options(v)
    random.Random(seeded_int(PERM_SEED, v['id'])).shuffle(opts)
    return [(LETTERS[j], i, b) for j, (i, b) in enumerate(opts)]


def prompt_text(v, cue, portrait=False):
    """A2 pilot.py wording. With portrait=True, the first sentence names the first image and the cue
    slot holds PORTRAIT_SENTENCE (plus any text cue). Empty cue leaves no double space."""
    lines = '\n'.join(f'{l}. {b}' for l, _, b in permuted_options(v))
    first = 'The first image' if portrait else 'The image'
    cue = ' '.join(s for s in ((PORTRAIT_SENTENCE if portrait else ''), cue) if s)
    head = (f'{first} shows five frames, in order from left to right, from an egocentric video recorded '
            'by the camera wearer.')
    middle = 'What should the camera wearer do next? Choose the most socially appropriate action.'
    return (' '.join(s for s in (head, cue, middle) if s) +
            f'\n\n{lines}\n\nRespond with only the letter of your choice and no explanation.')


def image_path(item_id):
    return DATA / 'model_previews' / f'{item_id}.jpg'


def make_splits(ann, table):
    """Split source videos (not items) with a seeded order: 50% train, 15% val, 35% test."""
    vids = sorted({video_of(k) for k in ann}, key=lambda x: seeded_int(SPLIT_SEED, x))
    n = len(vids)
    n_tr, n_va = round(n * SPLIT_FRACS[0]), round(n * SPLIT_FRACS[1])
    vsplit = {v: ('train' if i < n_tr else 'val' if i < n_tr + n_va else 'test') for i, v in enumerate(vids)}
    items = {s: [] for s in ('train', 'val', 'test')}
    for k in sorted(ann):
        items[vsplit[video_of(k)]].append(k)
    pool = {s: [k for k in ks if gold_nonempty(ann[k]) and table[k]['room_for_trait_cue'] == 'True']
            for s, ks in items.items()}
    clean = {s: [k for k in ks if gold_nonempty(ann[k])] for s, ks in items.items()}
    counts = {}
    for s in items:
        prim = sum(1 for k in pool[s] if table[k]['primary_norm'] == 'True')
        counts[s] = {'videos': sum(1 for v in vsplit.values() if v == s), 'items': len(items[s]),
                     'clean_items': len(clean[s]), 'pool_items': len(pool[s]),
                     'pool_primary_norm': prim, 'pool_secondary_only': len(pool[s]) - prim}
    return {'seed': SPLIT_SEED, 'fractions': SPLIT_FRACS, 'video_split': vsplit,
            'items': items, 'pool': pool, 'clean': clean, 'counts': counts}


def load_splits():
    return json.loads((HERE / 'results/splits.json').read_text())


def gate_items(splits, table, n=300):
    """Up to n train pool items: primary-norm items first, each group in seeded order."""
    pool = splits['pool']['train']
    order = lambda ks: sorted(ks, key=lambda k: seeded_int('gate-v1', k))
    prim = order([k for k in pool if table[k]['primary_norm'] == 'True'])
    rest = order([k for k in pool if table[k]['primary_norm'] != 'True'])
    return (prim + rest)[:n]


if __name__ == '__main__':
    ann, table = load_annotations(), load_item_table()
    sp = make_splits(ann, table)
    out = HERE / 'results/splits.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(sp, indent=1))
    print(json.dumps(sp['counts'], indent=2))
