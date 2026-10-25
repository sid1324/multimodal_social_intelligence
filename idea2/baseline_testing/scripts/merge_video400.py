"""Combine identical-protocol video predictions into the supplied 400-ID panel."""
import json
from pathlib import Path

from social_rl.common import ROOT, STORE, read_jsonl, sha256, write_json, write_jsonl


def main():
    manifest = STORE / 'data/manifests/ego_midterm_panel_accessible.jsonl'
    panel = read_jsonl(manifest)
    supplied = (ROOT / 'egonormia_400_sample_ids.txt').read_text().splitlines()
    if len(panel) != 400 or [r['id'] for r in panel] != supplied:
        raise ValueError('Panel differs from the supplied 400-ID order')
    if any(r['split'] != 'midterm' or r['dataset'] != 'ego' for r in panel):
        raise ValueError('Only the frozen EgoNormia midterm panel may be merged')
    provenance = {'panel_sha256': sha256(manifest), 'source_groups': len({r['group'] for r in panel}),
                  'models': {}}
    for kind in ('base', 'rloo', 'sft'):
        prior_dir = STORE / 'runs' / f'{kind}_ego'
        added_dir = STORE / 'runs' / f'{kind}_ego_video_remaining'
        prior_meta = json.loads((prior_dir / 'metadata.json').read_text())
        added_meta = json.loads((added_dir / 'metadata.json').read_text())
        if prior_meta['config'] != added_meta['config'] or prior_meta['adapter'] != added_meta['adapter']:
            raise ValueError(f'{kind}: backbone/adapter settings differ')
        if prior_meta['conditions'][0] != 'video' or added_meta['conditions'] != ['video']:
            raise ValueError(f'{kind}: video protocol differs')
        if 'max_new_tokens_override' in prior_meta or 'max_new_tokens_override' in added_meta:
            raise ValueError(f'{kind}: response caps differ')
        if prior_meta['ids'] not in (supplied[:140], supplied) or added_meta['ids'] != supplied[140:]:
            raise ValueError(f'{kind}: source run IDs differ from supplied order')
        first = [p for p in read_jsonl(prior_dir / 'predictions.jsonl') if p['condition'] == 'video']
        rest = read_jsonl(added_dir / 'predictions.jsonl')
        if [p['id'] for p in first] != supplied[:140] or [p['id'] for p in rest] != supplied[140:]:
            raise ValueError(f'{kind}: incomplete, reordered, or duplicate video predictions')
        if any(p['condition'] != 'video' or p['input_video_id'] != p['id'] for p in first + rest):
            raise ValueError(f'{kind}: non-matching video inputs')
        out = STORE / 'runs' / f'{kind}_ego_video400'
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / 'predictions.jsonl', first + rest)
        write_json(out / 'metadata.json', {
            'derivation': 'ordered video-only union; no new inference',
            'panel_sha256': sha256(manifest), 'config': prior_meta['config'],
            'adapter': prior_meta['adapter'], 'conditions': ['video'], 'ids': supplied,
            'sources': [str(prior_dir), str(added_dir)],
        })
        provenance['models'][kind] = {
            'n': len(first + rest), 'adapter': prior_meta['adapter'],
            'prior_prediction_sha256': sha256(prior_dir / 'predictions.jsonl'),
            'extension_prediction_sha256': sha256(added_dir / 'predictions.jsonl'),
            'merged_prediction_sha256': sha256(out / 'predictions.jsonl'),
        }
    results = ROOT / 'reports/results'
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / 'video400_provenance.json', provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
