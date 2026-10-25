"""Freeze adapted-model evaluation coverage to completed baseline comparisons."""
from social_rl.common import STORE, read_jsonl, write_jsonl

root = STORE/'data/manifests'
rows = read_jsonl(root/'ego_midterm_panel_accessible.jsonl')
preds = read_jsonl(STORE/'runs/base_ego/predictions.jsonl')
done = {}
for p in preds:
    done.setdefault(p['id'], set()).add(p['condition'])
matched = [r for r in rows if done.get(r['id'], set()) >= {'video','blind','mismatch'}]
if not matched:
    raise ValueError('No completed matched baseline examples')
write_jsonl(root/'ego_midterm_matched.jsonl', matched)
print('Frozen matched adapted-model panel:',len(matched))
