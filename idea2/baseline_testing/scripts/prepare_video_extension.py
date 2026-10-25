"""Freeze the remaining supplied EgoNormia IDs after the matched midterm run."""
from social_rl.common import STORE, read_jsonl, write_jsonl

root = STORE / 'data/manifests'
panel = read_jsonl(root / 'ego_midterm_panel_accessible.jsonl')
matched = read_jsonl(root / 'ego_midterm_matched.jsonl')
if len(panel) != 400 or len(matched) != 140:
    raise ValueError('Expected the frozen 400-ID panel and 140-ID matched run')
if [r['id'] for r in panel[:140]] != [r['id'] for r in matched]:
    raise ValueError('Matched run is not the first 140 supplied IDs')
remaining = panel[140:]
if len(remaining) != 260 or {r['id'] for r in remaining} & {r['id'] for r in matched}:
    raise ValueError('Invalid remaining-ID extension')
write_jsonl(root / 'ego_midterm_remaining260.jsonl', remaining)
print('Frozen correct-video extension:', len(remaining), 'remaining IDs')
