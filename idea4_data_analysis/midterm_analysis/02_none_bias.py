"""
The "None" bias: how often the model picks "None", where "None" sits among the options, and how
accurate the model is when it does commit to a real behavior.

Writes (in outputs/):
    <run>_none_bias.csv        per condition: picks None, mean p(None), accuracy when not picking None
    <run>_pred_positions.csv   per condition: % of predictions at each option position (1-5)

Usage:
    python 02_none_bias.py            # qwen3omni
    python 02_none_bias.py mimo
"""

import pandas as pd

from common import CONDS, LABEL, OUT_DIR, load, run_name


def main():
    run = run_name()
    rows = load(run)
    items = rows.drop_duplicates("id")

    print("WHERE 'None' SITS among the 5 options (number of items):")
    print("  " + ", ".join(f"position {int(k)}: {v}" for k, v in items.none_pos.value_counts().sort_index().items()))
    print(f"  the correct answer is 'None' for {items.correct_none.sum()} of {len(items)} items "
          f"({items.correct_none.mean():.1%})\n")

    out = []
    print("PICKING 'None' BY CONDITION")
    for c in CONDS:
        r = rows[rows.condition == c]
        committed = r[r.pred_none == 0]
        out.append({"condition": c, "picks_none": r.pred_none.mean(), "mean_p_none": r.p_none.mean(),
                    "n_committed": len(committed), "acc_when_committed": committed.is_correct.mean()})
        print(f"  {LABEL[c]:<14} picks None {r.pred_none.mean():5.1%}   mean p(None) {r.p_none.mean():.2f}   "
              f"when it picks a real behavior ({len(committed)} items): {committed.is_correct.mean():5.1%} correct")
    pd.DataFrame(out).to_csv(OUT_DIR / f"{run}_none_bias.csv", index=False)

    pos = pd.crosstab(rows.condition, rows.pred, normalize="index").reindex(CONDS) * 100
    pos.columns = [f"position_{int(c)}" for c in pos.columns]
    correct_pos = items.correct.value_counts(normalize=True).sort_index() * 100
    print("\nPREDICTED OPTION POSITION (% of predictions)")
    print(pos.round(1).to_string())
    print("  correct answers: " + ", ".join(f"{int(k)}: {v:.1f}" for k, v in correct_pos.items()))
    pos.to_csv(OUT_DIR / f"{run}_pred_positions.csv")

    # how adding audio changes "None" picks
    wide = rows.pivot(index="id", columns="condition", values="pred_none")
    stopped = int(((wide.video == 1) & (wide.both == 0)).sum())
    started = int(((wide.video == 0) & (wide.both == 1)).sum())
    print(f"\nVIDEO -> VIDEO + AUDIO: stopped picking None on {stopped} items, started on {started}")
    print(f"\nWrote {run}_none_bias.csv and {run}_pred_positions.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()