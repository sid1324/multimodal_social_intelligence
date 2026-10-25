"""
Report figures.

Writes (in outputs/):
    <run>_fig_overall.png/.pdf     (a) accuracy, 5 options  (b) accuracy, "None" removed  (c) % picking "None"
    <run>_fig_breakdown.png/.pdf   accuracy with "None" removed by norm category and by site

Usage:
    python 05_figures.py            # qwen3omni
    python 05_figures.py mimo
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CONDS, LABEL, OUT_DIR, load, run_name, wilson

# options-only is the neutral reference (gray); the three modality conditions use a
# colorblind-checked categorical set, plus a distinct marker shape each
COLOR = {"blind": "#8a8984", "video": "#2a78d6", "audio": "#eb6834", "both": "#1baf7a"}
MARKER = {"blind": "o", "video": "s", "audio": "^", "both": "D"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5df", "#ffffff"
MIN_CATEGORY_N = 40
SITE_NAMES = {"iiith": "IIIT Hyderabad", "frl_track_1_public": "Meta FRL", "nus": "NUS"}
OTHER_SITES = "Other (Bristol,\nIndiana, UTokyo)"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8, "axes.titlecolor": INK, "figure.facecolor": SURFACE})


def style(ax, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    (ax.xaxis if xgrid else ax.yaxis).grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_overall(rows, run):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2), dpi=300)
    panels = [("is_correct", "(a) Accuracy, 5 options (chance 20%)", 20, "Accuracy (%)"),
              ("correct_no_none", "(b) \"None\" removed (chance 25%)", 25, "Accuracy (%)"),
              ("pred_none", "(c) Picks \"None\"", None, "% of items")]
    x = np.arange(len(CONDS))
    for ax, (col, title, chance, ylab) in zip(axes, panels):
        vals, los, his = [], [], []
        for c in CONDS:
            v = rows.loc[rows.condition == c, col].dropna()
            lo, hi = wilson(v.sum(), len(v))
            vals.append(100 * v.mean()); los.append(100 * lo); his.append(100 * hi)
        vals, los, his = map(np.array, (vals, los, his))
        ax.bar(x, vals, width=0.62, color=[COLOR[c] for c in CONDS], edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.errorbar(x, vals, yerr=[vals - los, his - vals], fmt="none", ecolor=INK, elinewidth=0.8,
                    capsize=2, zorder=3)
        for xi, v, h in zip(x, vals, his):
            ax.text(xi, h + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        if chance:
            ax.axhline(chance, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[c].replace(" + ", "\n+ ").replace("Options only", "Options\nonly")
                            for c in CONDS], fontsize=6.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        style(ax)
    fig.tight_layout(w_pad=1.5)
    save(fig, f"{run}_fig_overall")


def dot_panel(ax, table, title, chance=25):
    ys = np.arange(len(table))[::-1]
    for y, (_, row) in zip(ys, table.iterrows()):
        vals = [row[c] for c in CONDS]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
    offsets = {"blind": 0.18, "video": 0.06, "audio": -0.06, "both": -0.18}
    for c in CONDS:
        ax.scatter(table[c], ys + offsets[c], s=22, marker=MARKER[c], color=COLOR[c],
                   edgecolor=SURFACE, linewidth=0.6, zorder=3, label=LABEL[c])
    ax.axvline(chance, color=MUTED, linestyle="--", linewidth=0.8, zorder=0, label="Chance (25%)")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{name}  (n={int(row.n)})" for name, row in table.iterrows()], fontsize=6.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy with \"None\" removed (%)")
    ax.set_title(title, loc="left")
    style(ax, xgrid=True)


def no_none_table(df, key):
    t = df.pivot_table(index=key, columns="condition", values="correct_no_none", aggfunc="mean")[CONDS] * 100
    t["n"] = df.drop_duplicates(["id", key]).groupby(key).size()
    return t.sort_values("n", ascending=False)


def fig_breakdown(rows, run):
    cat = no_none_table(rows.explode("categories"), "categories")
    cat = cat[cat.n >= MIN_CATEGORY_N]
    cat.index = [i.replace("/", "/\n") if len(i) > 14 else i for i in cat.index]
    site = no_none_table(rows.assign(site_name=rows.site.map(SITE_NAMES).fillna(OTHER_SITES)), "site_name")

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), dpi=300, gridspec_kw={"width_ratios": [1.15, 1]})
    dot_panel(axes[0], cat, "(a) By norm category")
    dot_panel(axes[1], site, "(b) By collection site")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2)
    save(fig, f"{run}_fig_breakdown")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}.png and {name}.pdf")


def main():
    run = run_name()
    rows = load(run)
    fig_overall(rows, run)
    fig_breakdown(rows, run)


if __name__ == "__main__":
    main()