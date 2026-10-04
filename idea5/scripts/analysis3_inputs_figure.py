"""Figure for Analysis 3: joint accuracy of six frozen models under four input combinations (clean-60, forced choice)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
REPORT = ROOT / 'output/idea5-report'
INK, SECONDARY, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
MODELS = ['Qwen3-VL-32B', 'GPT-5.5', 'GPT-5.6', 'GPT-6', 'Sonnet 5.5', 'Opus 5.5']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']  # fixed order, one per model
# (condition, row label), top to bottom
INPUTS = [('text_nn', 'options only'), ('dep_nn', '+ depth (no RGB)'), ('rgb_nn', '+ frames'),
          ('geo_nn', '+ frames, depth, segm.')]
# The repeated-frames control (rep_nn) is reported in the text only.


def load():
    key = json.loads((OUT / 'answer_key.json').read_text())
    answers = {}
    for path in (OUT / 'results/claude_single').glob('*_nn__*.json'):
        name, cond, _ = path.name.split('__')
        r = json.loads(path.read_text())
        answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'])
    for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
        for path in (OUT / 'results' / folder).glob('*_nn__*.json'):
            r = json.loads(path.read_text())
            answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
    for name in ['qwen32b_nn.json', 'qwen32b_controls.json', 'qwen32b_depth_only.json']:
        for r in json.loads((OUT / 'results' / name).read_text()):
            answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
    ids = list(key)

    def right(model, cond, item):
        a, j = answers[(model, cond, item)]
        k = key[item]
        return a is not None and k['action_order'][a - 1] == k['correct'] and k['justification_order'][j - 1] == k['correct']

    return {cond: np.array([[right(m, cond, i) for i in ids] for m in MODELS], float) for cond, _ in INPUTS}  # models x items


def main():
    R = load()
    rng = np.random.default_rng(42)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(3.25, 1.75))
    ax = fig.add_axes([0.385, 0.365, 0.47, 0.555])
    n = len(INPUTS)
    table = {}
    for row, (cond, label) in enumerate(INPUTS):
        y = n - 1 - row
        per_model = 100 * R[cond].mean(1)
        item_mean = R[cond].mean(0)
        boots = [100 * item_mean[rng.integers(0, len(item_mean), len(item_mean))].mean() for _ in range(5000)]
        mean, lo, hi = 100 * item_mean.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        table[cond] = {'mean': float(mean), 'ci95': [float(lo), float(hi)], 'per_model': dict(zip(MODELS, map(float, per_model)))}
        # The mean and its interval sit on their own lane below the model dots, so no dot is hidden behind it.
        ax.plot([lo, hi], [y - 0.27, y - 0.27], color=INK, lw=0.9, zorder=2)
        # Models sharing a value are spread vertically so every dot stays visible.
        order = np.argsort(per_model)
        offsets = np.zeros(len(MODELS))
        for rank, k in enumerate(order):
            close = [j for j in order[:rank] if abs(per_model[j] - per_model[k]) < 2.2]
            offsets[k] = 0.0 if not close else [0.2, 0.36, 0.1, 0.28][(len(close) - 1) % 4]
        for k in range(len(MODELS)):
            ax.plot(per_model[k], y + 0.08 + offsets[k], marker='o', ms=4.0, mfc=COLORS[k], mec='white', mew=0.6, ls='none', zorder=3)
        ax.plot(mean, y - 0.27, marker='D', ms=4.2, mfc=INK, mec='white', mew=0.6, ls='none', zorder=4)
        ax.text(1.04, y, f'{mean:.0f}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=6.8, color=INK)
    ax.text(1.04, n - 0.5, 'mean', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=6, color=SECONDARY)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label in INPUTS][::-1], fontsize=6.6)
    ax.set_xlim(20, 90)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_ylim(-0.62, n - 0.38)
    ax.set_xlabel('action and justification both correct (%)', fontsize=6.5, color=SECONDARY)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=SECONDARY, labelsize=6.5, width=0.6, length=2.5)
    ax.tick_params(axis='y', labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker='o', ms=4.0, mfc=COLORS[k], mec='white', mew=0.6, ls='none', label=m) for k, m in enumerate(MODELS)]
    handles.append(Line2D([], [], marker='D', ms=4.4, mfc=INK, mec='white', mew=0.6, color=INK, lw=0.9, label='six-model mean'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.9, labelspacing=0.3)
    for ext in ['pdf', 'png']:
        fig.savefig(REPORT / f'analysis3_inputs.{ext}', dpi=500, facecolor='white')
    (OUT / 'inputs_figure_values.json').write_text(json.dumps(table, indent=1))
    for cond, label in INPUTS:
        t = table[cond]
        print(f"{label:24s} mean {t['mean']:.1f} [{t['ci95'][0]:.1f}, {t['ci95'][1]:.1f}]  per model: " + ', '.join(f"{v:.0f}" for v in t['per_model'].values()))


if __name__ == '__main__':
    main()
