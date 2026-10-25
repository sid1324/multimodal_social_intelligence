"""Audit and merge the fixed 400-ID, 384-token EgoNormia diagnostic."""
import json

from social_rl.common import ROOT, STORE, read_jsonl, sha256, write_json, write_jsonl


def main():
    manifest = STORE / 'data/manifests/ego_midterm_panel_accessible.jsonl'
    panel = read_jsonl(manifest)
    ids = (ROOT / 'egonormia_400_sample_ids.txt').read_text().splitlines()
    if len(panel) != 400 or [r['id'] for r in panel] != ids:
        raise ValueError('Panel differs from supplied 400 IDs')
    provenance = {'panel_sha256': sha256(manifest), 'models': {}}
    for kind in ('base', 'rloo', 'sft'):
        ext_dir = STORE / 'runs' / f'{kind}_ego_cap384_extension'
        ext_meta = json.loads((ext_dir / 'metadata.json').read_text())
        ext = read_jsonl(ext_dir / 'predictions.jsonl')
        if ext_meta.get('max_new_tokens_override') != 384 or ext_meta['conditions'] != ['video']:
            raise ValueError(f'{kind}: incorrect 384-token video protocol')
        if kind == 'sft':
            if ext_meta['ids'] != ids or [p['id'] for p in ext] != ids:
                raise ValueError('SFT: incomplete or unordered predictions')
            combined, sources = ext, [ext_dir]
        else:
            prior_dir = STORE / 'runs' / f'{kind}_ego_cap384'
            prior_meta = json.loads((prior_dir / 'metadata.json').read_text())
            first = read_jsonl(prior_dir / 'predictions.jsonl')
            if prior_meta.get('max_new_tokens_override') != 384 or prior_meta['conditions'] != ['video']:
                raise ValueError(f'{kind}: original cap run has different protocol')
            if prior_meta['config'] != ext_meta['config'] or prior_meta['adapter'] != ext_meta['adapter']:
                raise ValueError(f'{kind}: original and extension model settings differ')
            if prior_meta['ids'] != ids[:40] or ext_meta['ids'] != ids[40:]:
                raise ValueError(f'{kind}: metadata IDs differ from supplied order')
            if [p['id'] for p in first] != ids[:40] or [p['id'] for p in ext] != ids[40:]:
                raise ValueError(f'{kind}: incomplete or unordered predictions')
            combined, sources = first + ext, [prior_dir, ext_dir]
        if any(p['condition'] != 'video' or p['input_video_id'] != p['id'] for p in combined):
            raise ValueError(f'{kind}: wrong input-video condition')
        out = STORE / 'runs' / f'{kind}_ego_cap384_video400'
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / 'predictions.jsonl', combined)
        write_json(out / 'metadata.json', {
            'derivation': 'ordered union of identical 384-token video protocol',
            'panel_sha256': sha256(manifest), 'config': ext_meta['config'],
            'adapter': ext_meta['adapter'], 'conditions': ['video'],
            'max_new_tokens_override': 384, 'ids': ids,
            'sources': [str(s) for s in sources],
        })
        provenance['models'][kind] = {
            'n': len(combined), 'adapter': ext_meta['adapter'],
            'source_prediction_sha256': {str(s): sha256(s / 'predictions.jsonl') for s in sources},
            'merged_prediction_sha256': sha256(out / 'predictions.jsonl'),
        }
    write_json(ROOT / 'reports/results/cap384_video400_provenance.json', provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
