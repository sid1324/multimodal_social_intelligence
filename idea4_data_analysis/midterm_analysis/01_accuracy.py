"""
Main results table: accuracy of each input condition, raw (5 options) and with "None" removed (4 options).

Writes (in outputs/):
    <run>_accuracy.csv        one row per condition
    <run>_paired_tests.csv    paired comparisons between conditions (exact McNemar)
    <run>_accuracy_table.tex  LaTeX table for the report (needs \\usepackage{booktabs})

Usage:
    python 01_accuracy.py            # qwen3omni
    python 01_accuracy.py mimo
"""

import pandas as pd

from common import CONDS, LABEL, OUT_DIR, load, mcnemar, run_name, wilson

COMPARISONS = [("blind", "video"), ("video", "both"), ("blind", "audio"), ("audio", "both")]


def main():
    run = run_name()
    rows = load(run)

    table = []
    for c in CONDS:
        r = rows[rows.condition == c]
        raw_k, raw_n = int(r.is_correct.sum()), len(r)
        nn = r.correct_no_none.dropna()
        nn_k, nn_n = int(nn.sum()), len(nn)
        raw_lo, raw_hi = wilson(raw_k, raw_n)
        nn_lo, nn_hi = wilson(nn_k, nn_n)
        table.append({"condition": c, "n": raw_n,
                      "acc_raw": raw_k / raw_n, "acc_raw_lo": raw_lo, "acc_raw_hi": raw_hi,
                      "acc_no_none": nn_k / nn_n, "acc_no_none_lo": nn_lo, "acc_no_none_hi": nn_hi,
                      "mean_p_correct": r.p_correct.mean(), "picks_none": r.pred_none.mean()})
    table = pd.DataFrame(table)
    table.to_csv(OUT_DIR / f"{run}_accuracy.csv", index=False)

    print("ACCURACY BY CONDITION   (raw chance 20%, 'None'-removed chance 25%)")
    for t in table.itertuples():
        print(f"  {LABEL[t.condition]:<14} raw {t.acc_raw:6.1%} [{t.acc_raw_lo:.1%}, {t.acc_raw_hi:.1%}]   "
              f"no None {t.acc_no_none:6.1%} [{t.acc_no_none_lo:.1%}, {t.acc_no_none_hi:.1%}]   "
              f"picks None {t.picks_none:5.1%}")

    tests = []
    print("\nPAIRED COMPARISONS (exact McNemar)")
    for metric, col in [("raw", "is_correct"), ("no None", "correct_no_none")]:
        wide = rows.pivot(index="id", columns="condition", values=col)[CONDS].dropna()
        for a, b in COMPARISONS:
            a_only, b_only, p = mcnemar(wide[a], wide[b])
            diff = wide[b].mean() - wide[a].mean()
            tests.append({"metric": metric, "from": a, "to": b, "diff": diff,
                          "right_only_from": a_only, "right_only_to": b_only, "p": p})
            print(f"  {metric:<7} {a:>5} -> {b:<5} {diff:+6.1%}   right only in {a}: {a_only:3d}, "
                  f"only in {b}: {b_only:3d}   p = {p:.2g}")
    pd.DataFrame(tests).to_csv(OUT_DIR / f"{run}_paired_tests.csv", index=False)

    write_latex(table, run)
    print(f"\nWrote {run}_accuracy.csv, {run}_paired_tests.csv, {run}_accuracy_table.tex in {OUT_DIR}")


def write_latex(table, run):
    def cell(v, lo, hi):
        return f"{100 * v:.1f} {{\\scriptsize [{100 * lo:.1f}, {100 * hi:.1f}]}}"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Input & Accuracy (5 options) & Accuracy (\\textit{None} removed) & Picks \\textit{None} \\\\",
        "\\midrule",
    ]
    for t in table.itertuples():
        lines.append(f"{LABEL[t.condition]} & {cell(t.acc_raw, t.acc_raw_lo, t.acc_raw_hi)} & "
                     f"{cell(t.acc_no_none, t.acc_no_none_lo, t.acc_no_none_hi)} & "
                     f"{100 * t.picks_none:.1f} \\\\")
    lines += [
        "\\midrule",
        "Chance & 20.0 & 25.0 & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Accuracy (\\%, 95\\% CI) of {run} on {table.n.iloc[0]} EgoNormia items by input. "
        "\\textit{None} removed: the best of the four real behaviors, ignoring the \\textit{None} option.}",
        f"\\label{{tab:{run}_accuracy}}",
        "\\end{table}",
    ]
    (OUT_DIR / f"{run}_accuracy_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()