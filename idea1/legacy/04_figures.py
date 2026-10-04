"""04 - Figures for Analysis 1 (reads only outputs/tables/*.csv)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "outputs", "tables"); F = os.path.join(ROOT, "outputs", "figures")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
COL = {"blind": "#bdbdbd", "desc": "#6baed6", "grid": "#08519c", "frames": "#fd8d3c", "video": "#d94801"}
NAME = {"blind": "Options only", "desc": "Text description", "grid": "Frame-grid image",
        "frames": "5 discrete frames", "video": "Native video"}

ov = pd.read_csv(os.path.join(T, "ladder_overall.csv"))
ov = ov[ov.tier == "primary"]
pr = pd.read_csv(os.path.join(T, "ladder_pairwise.csv"))
pdlt = pd.read_csv(os.path.join(T, "ladder_by_norm_pooled_delta.csv"))

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2), gridspec_kw={"width_ratios": [1.15, 1]})
# Panel a: ladder
ax = axes[0]
fams = list(dict.fromkeys(ov.family))
w = 0.26
for i, c in enumerate(["blind", "desc", "grid"]):
    s = ov[ov.condition == c].set_index("family").loc[fams]
    x = np.arange(len(fams)) + (i - 1) * w
    ax.bar(x, 100 * s.both, w, color=COL[c], label=NAME[c],
           yerr=[100 * (s.both - s.both_lo), 100 * (s.both_hi - s.both)], capsize=1.5, error_kw={"lw": 0.6})
ax.set_xticks(np.arange(len(fams))); ax.set_xticklabels([f"{f}\n(n={int(ov[ov.family==f].n_items.iloc[0])})" for f in fams])
ax.set_ylabel("Joint action+justification acc. (%)"); ax.set_ylim(0, 60)
ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=3)
ax.set_title("(a) Same model, same items, richer perceptual input", fontsize=7)
# Panel b: per-norm pooled deltas
ax = axes[1]
cats = [c for c in pdlt.category.unique() if c != "ALL"]
short = {"Coordination/Proactivity": "Coordination", "Communication/Legibility": "Communication"}
d2 = pdlt[pdlt.step == "desc->grid"].set_index("category").loc[cats]
d1 = pdlt[pdlt.step == "blind->desc"].set_index("category").loc[cats]
order = d2.delta.sort_values().index
y = np.arange(len(order))
ax.barh(y + 0.18, 100 * d2.loc[order, "delta"], 0.36, color=COL["grid"], label="desc. \u2192 image",
        xerr=[100 * (d2.loc[order, "delta"] - d2.loc[order, "lo"]), 100 * (d2.loc[order, "hi"] - d2.loc[order, "delta"])],
        capsize=1.5, error_kw={"lw": 0.6})
ax.barh(y - 0.18, 100 * d1.loc[order, "delta"], 0.36, color=COL["desc"], label="options \u2192 desc.",
        xerr=[100 * (d1.loc[order, "delta"] - d1.loc[order, "lo"]), 100 * (d1.loc[order, "hi"] - d1.loc[order, "delta"])],
        capsize=1.5, error_kw={"lw": 0.6})
ax.set_yticks(y); ax.set_yticklabels([short.get(c, c) for c in order])
ax.set_xlabel("Gain in joint acc. (pts), avg. of 3 families")
ax.axvline(0, color="k", lw=0.5); ax.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1)
ax.set_title("(b) Gain by norm of the correct action", fontsize=7); ax.set_xlim(0, 38)
plt.tight_layout()
plt.savefig(os.path.join(F, "fig_ladder.pdf")); plt.savefig(os.path.join(F, "fig_ladder.png"), dpi=300)

# Static vs dynamic subset figure (supplementary)
sd = pr[pr.to.isin(["frames", "video"])]
fig, ax = plt.subplots(figsize=(3.25, 1.9))
lab = [f"{r.family}\n{NAME[r.to]} (n={r.n_items})" for r in sd.itertuples()]
y = np.arange(len(sd))
ax.barh(y, 100 * sd.delta, color=[COL[t] for t in sd.to],
        xerr=[100 * (sd.delta - sd.delta_lo), 100 * (sd.delta_hi - sd.delta)], capsize=1.5, error_kw={"lw": 0.6})
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Joint acc. relative to frame-grid image (pts)")
plt.tight_layout(); plt.savefig(os.path.join(F, "fig_static_vs_dynamic.pdf")); plt.savefig(os.path.join(F, "fig_static_vs_dynamic.png"), dpi=300)
print("figures written")
