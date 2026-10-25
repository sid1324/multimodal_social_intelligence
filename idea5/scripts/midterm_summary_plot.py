"""Forest plot of every spatial -> social contrast in Idea 5 (joint-accuracy difference, percentage points, 95% CI).

Item-level CIs come from a bootstrap that resamples items (pooled direct-eval rows resample an item with all of its
models; low-rank CV resamples source videos). Fine-tuning rows use the CIs in lora_pilot/scored_followup.json; RL
rows are added from rl_pilot/scored.json when it exists.
Writes output/idea5-midterm/figures/summary_forest.{png,pdf} and summary_forest.json.

usage: midterm_summary_plot.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
RES = MID / 'results'
DIRECT = MID / 'direct_eval'
FIG = MID / 'figures'
B = 5000
INK, MUTED, GRID = '#1f2328', '#6e7781', '#d0d7de'


def boot(diff_by_cluster, seed=0):
    """diff_by_cluster: {cluster: list of per-row differences}. Mean over rows, resampling clusters."""
    clusters = list(diff_by_cluster)
    sums = np.array([sum(diff_by_cluster[c]) for c in clusters])
    counts = np.array([len(diff_by_cluster[c]) for c in clusters])
    rng = np.random.default_rng(seed)
    draws = [sums[s].sum() / counts[s].sum() for s in (rng.integers(0, len(clusters), len(clusters)) for _ in range(B))]
    return 100 * sums.sum() / counts.sum(), 100 * np.percentile(draws, 2.5), 100 * np.percentile(draws, 97.5)


def joint(v):
    return int(v['action'] == v['gold'] and v['justification'] == v['gold'])


def embedding(a, b, variant, part):
    A, Bv = (json.loads((RES / f'{c}_{variant}.json').read_text())[part] for c in (a, b))
    groups = defaultdict(list)
    for i in set(A) & set(Bv):
        groups[i.split('_')[0] if part == 'cv' else i].append(joint(A[i]) - joint(Bv[i]))
    return boot(groups)


def direct_rows(model, cond):
    key = json.loads((MID / 'lora_pilot/answer_key.json').read_text())
    out = {}
    for f in (DIRECT / model / cond).glob('*.json'):
        r = json.loads(f.read_text())
        k = key[r['id']]
        a = k['action_order'][r['action_position'] - 1] if r['action_position'] else None
        j = k['justification_order'][r['justification_position'] - 1] if r['justification_position'] else None
        out[r['id']] = int(a == k['gold'] and j == k['gold'])
    return out


def averaged(model, cond):
    """Mean correctness over the original run and its '<cond>_rep' repeat when a full repeat exists."""
    rows, rep = direct_rows(model, cond), direct_rows(model, cond + '_rep') if (DIRECT / model / f'{cond}_rep').exists() else {}
    if len(rep) < 150:
        return rows
    return {i: (rows[i] + rep[i]) / 2 for i in rows if i in rep}


def direct(a, b, models):
    groups = defaultdict(list)
    for m in models:
        A, Bv = averaged(m, a), averaged(m, b)
        for i in set(A) & set(Bv):
            groups[i].append(A[i] - Bv[i])
    return boot(groups)


def main():
    rows = []  # (group, label, est, lo, hi)
    for group, variant, part in (('Embedding, zero-shot (n=199)', 'full', 'zero_shot'),
                                 ('Embedding, low-rank scorer, CV (n=799)', 'lowrank', 'cv')):
        for label, a, b in [('Masks + descriptions vs RGB + description (C5-C2)', 'C5', 'C2'),
                            ('Correct vs mismatched masks (C5-X1)', 'C5', 'X1'),
                            ('Masks vs RGB repeated (C5-X3)', 'C5', 'X3')]:
            rows.append((group, label, *embedding(a, b, variant, part)))
    models = ['gemini', 'gpt', 'qwen32']
    for label, a, b in [('+ maps vs RGB', 'maps', 'rgb'), ('Correct vs mismatched maps', 'maps', 'maps_wrong'),
                        ('+ maps vs RGB repeated', 'maps', 'rgb_x4'), ('+ descriptions vs RGB', 'desc', 'rgb'),
                        ('Correct vs mismatched descriptions', 'desc', 'desc_wrong')]:
        rows.append(('Direct answering, Gemini 4 + GPT-5.6 + Qwen3-VL-32B', label, *direct(a, b, models)))
    for m, name in (('gpt', 'GPT-5.6'), ('gemini', 'Gemini 4'), ('qwen32', 'Qwen3-VL-32B')):
        if len(direct_rows(m, 'desc_only_flip')) >= 150:
            # the reworded copy went through the same rewrite as the reversed one, so it is the closer arm
            base = 'desc_only_para' if len(direct_rows(m, 'desc_only_para')) >= 150 else 'desc_only'
            rows.append(('Counterfactual: spatial facts reversed (text only)',
                         f'{name}: {"reworded" if base != "desc_only" else "original"} vs reversed', *direct(base, 'desc_only_flip', [m])))
    forced = [m for m in models if len(direct_rows(m, 'maps_wrong_forced')) >= 150]
    if forced:
        group = 'Forced choice, "None" removed (GPT-5.6 + Qwen3-VL-32B)'
        rows.append((group, '+ maps vs RGB', *direct('maps_forced', 'rgb_forced', forced)))
        rows.append((group, 'Correct vs mismatched maps', *direct('maps_forced', 'maps_wrong_forced', forced)))
    ctx_models = [m for m in models if len(direct_rows(m, 'rgb_ctx_wrong')) >= 150]
    if ctx_models:
        group = f"Non-spatial text control: dataset scene description ({', '.join(m.replace('qwen32', 'Qwen-32B').replace('gpt', 'GPT').replace('gemini', 'Gemini') for m in ctx_models)})"
        rows.append((group, '+ scene description vs RGB', *direct('rgb_ctx', 'rgb', ctx_models)))
        rows.append((group, 'Correct vs mismatched scene description', *direct('rgb_ctx', 'rgb_ctx_wrong', ctx_models)))
    for m, name in (('gpt', 'GPT-5.6 (n=200)'), ('qwen32', 'Qwen3-VL-32B (n=100)')):
        if len(direct_rows(m, 'rgb_spcot')) >= 75:
            rows.append(('Prompting: reason about space first', f'{name}: spatial analysis vs generic step-by-step',
                         *direct('rgb_spcot', 'rgb_cot', [m])))
    paper = MID / 'paper_models.json'  # written by the published-checkpoint comparison
    if paper.exists():
        names = {'spatialladder3b-qwen25vl3b': 'SpatialLadder-3B vs Qwen2.5-VL-3B (paper: +23.5 spatial)',
                 'mindcube3b-qwen25vl3b': 'MindCube-RL-3B vs Qwen2.5-VL-3B (paper: +23.5 MindCube)',
                 'spacer7b-qwen25vl7b': 'SpaceR-7B vs Qwen2.5-VL-7B (paper: +11.2 VSI-Bench)',
                 'spatialssrl7b-qwen25vl7b': 'Spatial-SSRL-7B vs Qwen2.5-VL-7B (paper: +3.9 avg)'}
        res = json.loads(paper.read_text())
        for k, label in names.items():
            c = res.get(f'{k}:rgb')
            if c:
                rows.append(('Published spatial fine-tunes vs their base (n=200)', label, c['diff'], c['lo'], c['hi']))
    ft = json.loads((MID / 'lora_pilot/scored_followup.json').read_text())['contrasts']['spatial-control']['joint']
    rows.append(('Fine-tuning Qwen3-VL-8B (2 seeds, n=400)', 'Spatial LoRA vs non-spatial LoRA',
                 100 * ft['diff'], 100 * ft['ci95'][0], 100 * ft['ci95'][1]))
    combined = MID / 'lora_pilot/scored_combined.json'  # spatial LoRA x (RGB | RGB + maps + descriptions), all cells on one pod
    if combined.exists():
        c2 = json.loads(combined.read_text())['contrasts']['pod']
        for k, label in (('a_spatial_combined-base_rgb', 'Spatial LoRA + maps & descriptions vs base + RGB'),
                         ('b_base_combined-base_rgb', 'Base: + maps & descriptions vs RGB'),
                         ('c_spatial_combined-base_combined', 'Spatial LoRA vs base, both given maps & descriptions')):
            c = c2[k]
            rows.append(('Spatial LoRA x combined input, Qwen3-VL-8B (n=200)', label, 100 * c['diff'], 100 * c['ci95'][0], 100 * c['ci95'][1]))
    names = {'R0-base_social': 'Norm reward vs untrained (reference)', 'base_spatial-base_social': 'Untrained: spatial-first prompt vs plain',
             'R1-R0': 'Spatial + norm reward vs norm reward', 'R2-R0': 'Spatial format, norm reward vs norm reward',
             'R1-R2': 'Spatial reward vs spatial format only', 'R3-base_spatial': 'Spatial-only reward vs untrained'}
    # 90-step runs give the reference gain; the spatial arms are compared with R0 at matched 40 steps.
    for path, group, keep in ((MID / 'rl_pilot/scored.json', 'RL (GRPO) Qwen3-VL-8B, 90 steps', ('R0-base_social', 'base_spatial-base_social')),
                              (MID / 'rl_pilot/matched40/scored.json', 'RL (GRPO) Qwen3-VL-8B, matched 40 steps',
                               ('R1-R0', 'R2-R0', 'R1-R2', 'R3-base_spatial'))):
        if path.exists():
            for name, c in json.loads(path.read_text())['contrasts'].items():
                if name in keep:
                    rows.append((group, names[name], 100 * c['diff'], 100 * c['ci95'][0], 100 * c['ci95'][1]))
    meta = MID / 'meta.json'
    if meta.exists():
        pooled = json.loads(meta.read_text())['pooled']
        for key, label in (('all', f"All {pooled['all']['k']} spatial-specific contrasts"),
                           ('need_high_minus_low', 'Effect on high- minus low-spatial-need items')):
            p = pooled[key]
            rows.append(('Pooled (random effects, midterm_meta.py)', label, p['est'], p['lo'], p['hi']))

    FIG.mkdir(exist_ok=True)
    (FIG / 'summary_forest.json').write_text(json.dumps(rows, indent=1))
    labels, ys, headers, y, last = [], [], [], 0, None
    fig_h = 0.32 * (len(rows) + len({r[0] for r in rows})) + 0.8
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    for group, label, est, lo, hi in rows:
        if group != last:
            labels.append(group)
            ys.append(y)
            headers.append(len(labels) - 1)
            y += 1
            last = group
        ax.plot([lo, hi], [y, y], color=INK, lw=2, solid_capstyle='round')
        ax.plot(est, y, 'o', ms=6, color=INK, mec='white', mew=1.5, zorder=3)
        ax.text(max(hi, 0) + 0.6, y, f'{est:+.1f} [{lo:+.1f}, {hi:+.1f}]', va='center', fontsize=7.5, color=MUTED)
        labels.append('    ' + label)
        ys.append(y)
        y += 1
    ax.axvline(0, color=MUTED, lw=1, zorder=0)
    ax.set_yticks(ys, labels, fontsize=8, color=INK)
    for n in headers:
        ax.get_yticklabels()[n].set_fontweight('bold')
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlabel('Change in joint accuracy (percentage points, 95% CI)', fontsize=8.5, color=INK)
    ax.tick_params(axis='x', labelsize=8, colors=MUTED)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='y', length=0)
    lo_all = min(r[3] for r in rows)
    hi_all = max(r[4] for r in rows)
    ax.set_xlim(lo_all - 2, hi_all + 12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'summary_forest.{ext}', dpi=200, bbox_inches='tight')
    for r in rows:
        print(f'{r[0][:40]:40s} | {r[1][:45]:45s} {r[2]:+5.1f} [{r[3]:+5.1f}, {r[4]:+5.1f}]')


if __name__ == '__main__':
    main()
