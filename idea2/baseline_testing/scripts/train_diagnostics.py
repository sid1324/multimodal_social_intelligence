"""Summarize saved RLOO events without treating reward as EgoNormia evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from social_rl.common import config, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = {r['id']: r for r in read_jsonl(args.manifest)}
    if any(r['dataset'] != 'r3' or r['split'] != 'train' for r in manifest.values()):
        raise ValueError('Diagnostics require the R3 training manifest')
    events = read_jsonl(Path(args.run) / 'train_log.jsonl')
    if not events or any(e['method'] != 'rloo' for e in events):
        raise ValueError('No RLOO events')
    unknown = {e['id'] for e in events} - set(manifest)
    if unknown:
        raise ValueError(f'Trace contains IDs outside R3 training: {sorted(unknown)[:3]}')
    rewards = np.asarray([np.mean(e['rewards']) for e in events])
    updates = np.asarray([int('step_after' in e) for e in events])
    steps = updates.cumsum()
    lengths = [length for e in events for length in e.get('completion_lengths', [])]
    kl = [e['sampled_kl'] for e in events if 'sampled_kl' in e]
    result = {
        'sampled_groups': len(events),
        'optimizer_updates': int(updates.sum()),
        'unique_questions': len({e['id'] for e in events}),
        'unique_source_videos': len({e['video_id'] for e in events}),
        'unique_movie_groups': len({manifest[e['id']]['group'] for e in events}),
        'sampled_correctness': float(rewards.mean()),
        'uniform_reward_skips': sum(e.get('skipped') == 'zero_reward_variation' for e in events),
        'nonfinite_gradient_skips': sum(e.get('skipped') == 'nonfinite_gradients' for e in events),
        'completion_cap_hits': sum(n >= config()['max_new_tokens'] for n in lengths),
        'sampled_completions': len(lengths),
        'mean_completion_tokens': float(np.mean(lengths)),
        'mean_sampled_kl_on_update_candidates': float(np.mean(kl)) if kl else None,
        'max_sampled_kl': float(max(kl)) if kl else None,
        'sum_group_seconds': float(sum(e['seconds'] for e in events)),
        'max_peak_reserved_gb': float(max(e.get('peak_reserved_gb', 0) for e in events)),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'train_diagnostics.json', result)
    x = np.arange(1, len(events) + 1)
    width = min(10, len(events))
    moving = np.convolve(rewards, np.ones(width) / width, mode='valid')
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout='constrained')
    axes[0].plot(x, steps, color='#20639b')
    axes[0].set_ylabel('Optimizer updates')
    axes[0].grid(alpha=.2)
    axes[1].plot(x[width - 1:], moving, color='#dd6b20', label=f'{width}-group mean')
    axes[1].set(xlabel='Sampled R3 group', ylabel='Correct rollouts / 4', ylim=(0, 1))
    axes[1].grid(alpha=.2)
    fig.suptitle('R3-FDT RLOO pilot: training trace (not benchmark performance)')
    fig.savefig(out / 'train_trace.png', dpi=180)
    plt.close(fig)
    print(result)


if __name__ == '__main__':
    main()
