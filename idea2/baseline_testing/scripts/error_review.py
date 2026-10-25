"""Build a review workbook with actual sampled frames; no invented error causes."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from social_rl.common import config, read_jsonl, stable_hash, write_jsonl
from social_rl.model import sample_video
from social_rl.tasks import item_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    records = read_jsonl(args.manifest)
    if any(r.get('split') == 'final' for r in records):
        raise ValueError('Final evaluation partition is reserved')
    rows = {r['id']: r for r in records}
    preds = read_jsonl(Path(args.run)/'predictions.jsonl')
    video = {p['id']: p for p in preds if p['condition'] == 'video'}
    errors = [i for i,p in video.items() if not item_metrics(rows[i],p).get('joint',0)]
    successes = [i for i,p in video.items() if item_metrics(rows[i],p).get('joint',0)]
    selection = sorted(errors,key=stable_hash)[:25] + sorted(successes,key=stable_hash)[:5]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    workbook = []
    page = ['<!doctype html><meta charset="utf-8"><title>Midterm error review</title>',
            '<style>body{font:16px sans-serif;max-width:1200px;margin:2rem auto}img{width:100%}pre{white-space:pre-wrap}article{border-bottom:2px solid #888;padding:1rem}</style>',
            '<h1>Midterm evidence review</h1><p>Frames are the model input. Failure causes require analyst inspection; model explanations are not verified evidence.</p>']
    for uid in selection:
        row = rows[uid]
        frames, meta = sample_video(row['video_path'], config()['frames'], config()['frame_max_side'])
        fig, axes = plt.subplots(2,4,figsize=(12,5))
        for ax,frame,index in zip(axes.flat,frames,meta['frames_indices']):
            ax.imshow(frame)
            ax.set_title(f'{index/meta["fps"]:.1f}s')
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(out/f'{uid}.jpg',dpi=130)
        plt.close(fig)
        related = {p['condition']:p for p in preds if p['id']==uid}
        entry = {'id':uid,'group':row['group'],'correct_action':row['answer'], 'sensible':row['sensible'],
                 'behaviors':row['behaviors'],'justifications':row['justifications'],
                 'predictions':related, 'taxonomy':row['taxonomy'],
                 'analyst_failure_type':None,'analyst_evidence':None,'improvement_hypothesis':None}
        workbook.append(entry)
        page += ['<article><h2>'+html.escape(uid)+'</h2>',f'<img src="{uid}.jpg">',
                 '<pre>'+html.escape(json.dumps(entry,indent=2,ensure_ascii=False))+'</pre></article>']
    # Never overwrite completed annotations: machine-generated workbook has a separate name.
    write_jsonl(out/'review_unannotated.jsonl',workbook)
    (out/'index.html').write_text('\n'.join(page))
    print(f'Prepared {len(workbook)} cases ({min(25,len(errors))} errors, {min(5,len(successes))} successes).')

if __name__ == '__main__':
    main()
