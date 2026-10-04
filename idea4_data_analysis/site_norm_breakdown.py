"""
Norm-by-site breakdown for the audio-available clips, plus the audio-coverage charts.

Reads p2_data_analysis/egonormia_audio_check.csv (from check_audio.py).

Prints:
    (1) norm composition of each collection site (audio-covered items, all categories)
    (2) the three exclusive groups used in the visual probe (Communication / Safety / Proxemics)
        by site, with a chi-square test of whether group mix depends on site
    (3) within-site audio coverage by category, for sites that only partly have audio
        (is the Safety / Coordination gap a site effect or does it hold inside a site?)

Writes (in p2_data_analysis/):
    site_norm_counts.csv          site x category counts (audio-covered items)
    site_group_counts.csv         site x exclusive-group counts (visual-probe items)
    audio_coverage.png            analysis 1 figure: coverage by norm category and by site
    site_group_composition.png    exclusive-group mix per site (confound for the visual probe)

Usage:
    python site_norm_breakdown.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_CHECK_CSV = SCRIPT_DIR / "egonormia_audio_check.csv"

MIN_SITE_ITEMS = 10          # sites smaller than this are folded into "other" in tables/tests
PARTIAL_SITE_RANGE = (0.10, 0.90)  # "partly has audio" = coverage between these
PARTIAL_SITE_MIN_ITEMS = 30

GROUPS = ["Communication/Legibility", "Safety", "Proxemics"]
GROUP_COLORS = {"Communication/Legibility": "#2a78d6", "Safety": "#eb6834", "Proxemics": "#1baf7a"}
BAR_COLOR = "#2a78d6"
INK, MUTED, GRID = "#2b2b29", "#6b6a65", "#e6e5df"


def load():
    if not AUDIO_CHECK_CSV.exists():
        sys.exit(f"{AUDIO_CHECK_CSV.name} not found. Run check_audio.py first.")
    df = pd.read_csv(AUDIO_CHECK_CSV, keep_default_na=False)
    df["covered"] = df["audio_covers_window"].astype(str) == "True"
    df["cats"] = df["correct_norm_categories"].apply(lambda s: [c for c in str(s).split("|") if c])
    df["site"] = df["video_source"].replace("", "(unknown)")
    return df


def exclusive_group(cats):
    hits = [g for g in GROUPS if g in cats]
    return hits[0] if len(hits) == 1 else None


def fold_small_sites(site_series):
    counts = site_series.value_counts()
    small = counts[counts < MIN_SITE_ITEMS].index
    return site_series.where(~site_series.isin(small), "other (<%d items)" % MIN_SITE_ITEMS)


def section1(df):
    cov = df[df["covered"]].copy()
    cov["site_f"] = fold_small_sites(cov["site"])
    long = cov[["id", "site_f", "cats"]].explode("cats").dropna(subset=["cats"]).reset_index(drop=True)
    counts = pd.crosstab(long["site_f"], long["cats"])
    n_items = cov.groupby("site_f").size()
    counts = counts.loc[n_items.sort_values(ascending=False).index]
    counts.insert(0, "n_items", n_items)
    counts.to_csv(SCRIPT_DIR / "site_norm_counts.csv")

    share = counts.drop(columns="n_items").div(counts["n_items"], axis=0)
    order = counts.drop(columns="n_items").sum().sort_values(ascending=False).index
    overall = long["cats"].value_counts().reindex(order) / len(cov)

    print(f"(1) NORM COMPOSITION BY SITE (audio-covered items, n={len(cov)})")
    print("    % of a site's items whose correct answer carries the category (rows can exceed 100%)\n")
    short = {c: c.split("/")[0][:11] for c in order}
    header = f"    {'site':<22}{'n':>5} " + "".join(f"{short[c]:>12}" for c in order)
    print(header)
    for site, row in share[order].iterrows():
        print(f"    {site:<22}{counts.loc[site, 'n_items']:>5} " + "".join(f"{v:>12.0%}" for v in row))
    print(f"    {'ALL':<22}{len(cov):>5} " + "".join(f"{v:>12.0%}" for v in overall))


def section2(df):
    from scipy.stats import chi2_contingency

    cov = df[df["covered"]].copy()
    cov["group"] = cov["cats"].apply(exclusive_group)
    g = cov.dropna(subset=["group"]).copy()
    g["site_f"] = fold_small_sites(g["site"])
    table = pd.crosstab(g["site_f"], g["group"]).reindex(columns=GROUPS, fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table.to_csv(SCRIPT_DIR / "site_group_counts.csv")

    print(f"\n(2) VISUAL-PROBE GROUPS BY SITE (exclusive groups, n={len(g)})")
    pct = table.div(table.sum(axis=1), axis=0)
    print(f"    {'site':<22}{'n':>5}" + "".join(f"{x.split('/')[0][:13]:>15}" for x in GROUPS))
    for site in table.index:
        print(f"    {site:<22}{table.loc[site].sum():>5}" +
              "".join(f"{table.loc[site, x]:>7} ({pct.loc[site, x]:>4.0%})" for x in GROUPS))
    tot = table.sum()
    print(f"    {'ALL':<22}{tot.sum():>5}" + "".join(f"{tot[x]:>7} ({tot[x] / tot.sum():>4.0%})" for x in GROUPS))

    test = table[table.sum(axis=1) >= MIN_SITE_ITEMS]
    chi2, p, dof, _ = chi2_contingency(test)
    n = test.values.sum()
    v = (chi2 / (n * (min(test.shape) - 1))) ** 0.5
    print(f"\n    Does group mix depend on site?  chi2 = {chi2:.1f}, dof = {dof}, p = {p:.2g}, "
          f"Cramer's V = {v:.2f}")
    print("    (V ~0.1 small, ~0.3 medium, ~0.5 large association)")
    return table


def section3(df):
    print("\n(3) WITHIN-SITE AUDIO COVERAGE BY CATEGORY (sites that only partly have audio)")
    site_cov = df.groupby("site")["covered"].agg(["mean", "size"])
    lo, hi = PARTIAL_SITE_RANGE
    partial = site_cov[(site_cov["mean"] > lo) & (site_cov["mean"] < hi) &
                       (site_cov["size"] >= PARTIAL_SITE_MIN_ITEMS)].sort_values("size", ascending=False)
    if partial.empty:
        print("    (no site qualifies)")
        return
    long = df.explode("cats").dropna(subset=["cats"]).reset_index(drop=True)
    for site, r in partial.iterrows():
        sub = long[long["site"] == site]
        t = sub.groupby("cats")["covered"].agg(["sum", "size"]).sort_values("size", ascending=False)
        print(f"\n    site = {site}: {r['mean']:.0%} of {int(r['size'])} items have audio")
        for cat, rr in t.iterrows():
            print(f"      {cat:<28}{int(rr['sum']):>4} / {int(rr['size']):<4} ({rr['sum'] / rr['size']:>4.0%})")


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="both", labelsize=7, colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_coverage(df):
    long = df.explode("cats").reset_index(drop=True)
    long["cats"] = long["cats"].fillna("(none)")
    by_cat = long.groupby("cats")["covered"].agg(["mean", "size"])
    by_cat = by_cat[by_cat["size"] >= 5].sort_values("mean")
    by_site = df.groupby("site")["covered"].agg(["mean", "size"])
    by_site = by_site[by_site["size"] >= 5].sort_values("mean")
    overall = df["covered"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    for ax, data, title in [(axes[0], by_cat, "By norm category (correct answer)"),
                            (axes[1], by_site, "By collection site")]:
        ys = range(len(data))
        ax.barh(list(ys), data["mean"] * 100, color=BAR_COLOR, height=0.62)
        for y, (m, n) in zip(ys, data[["mean", "size"]].itertuples(index=False)):
            ax.text(m * 100 + 1.5, y, f"{m:.0%}", va="center", fontsize=6.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=3)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"{i}  (n={int(n)})" for i, n in zip(data.index, data["size"])], fontsize=7, color=INK)
        ax.axvline(overall * 100, color="#8a8984", linestyle="--", linewidth=1)
        ax.text(overall * 100, -0.75, f" overall {overall:.0%}", fontsize=6.5, color="#8a8984", va="center")
        ax.set_ylim(-1.1, len(data) - 0.4)
        ax.set_xlim(0, 112)
        ax.set_xlabel("% of items with audio covering the clip window", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8.5, color=INK, pad=8)
        style(ax)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "audio_coverage.png", bbox_inches="tight")


def plot_composition(table):
    pct = table.div(table.sum(axis=1), axis=0) * 100
    pct = pct.iloc[::-1]  # largest site on top
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(pct) + 1.2), dpi=200)
    left = pd.Series(0.0, index=pct.index)
    for g in GROUPS:
        ax.barh(range(len(pct)), pct[g], left=left, color=GROUP_COLORS[g], height=0.62,
                edgecolor="white", linewidth=1.5, label=g)
        for y, (l, w) in enumerate(zip(left, pct[g])):
            if w >= 9:
                ax.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=6.5, color="white")
        left += pct[g]
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels([f"{s}  (n={table.loc[s].sum()})" for s in pct.index], fontsize=7, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of site's visual-probe items", fontsize=7, color=MUTED)
    ax.set_title("Norm-group mix by collection site (audio-covered, exclusive groups)", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    style(ax)
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "site_group_composition.png", bbox_inches="tight")


def main():
    df = load()
    section1(df)
    table = section2(df)
    section3(df)
    plot_coverage(df)
    plot_composition(table)
    print("\nWrote site_norm_counts.csv, site_group_counts.csv, audio_coverage.png, site_group_composition.png")


if __name__ == "__main__":
    main()