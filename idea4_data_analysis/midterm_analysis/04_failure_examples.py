"""
Sort items into success/failure types for the qualitative error analysis.

Types (an item can have more than one):
    A_audio_made_it_commit    video picks None; video + audio picks the correct behavior
    B_audio_misled            video is correct; video + audio picks a wrong real behavior
    C_audio_only_informative  with None removed: audio correct, but options-only and video both wrong
    D_confident_wrong_video   video and video + audio both pick the same wrong real behavior with p > 0.6

Writes (in outputs/):
    <run>_failure_examples.csv   one row per item and type, with options, predictions and p(correct)

Usage:
    python 04_failure_examples.py            # qwen3omni
    python 04_failure_examples.py mimo
"""

import pandas as pd

from common import CONDS, OUT_DIR, load, run_name

CONFIDENT = 0.6
SHOW_PER_TYPE = 2


def main():
    run = run_name()
    rows = load(run)
    by = {c: rows[rows.condition == c].set_index("id") for c in CONDS}

    out = []
    for i in by["both"].index:
        b, v, a, o = by["both"].loc[i], by["video"].loc[i], by["audio"].loc[i], by["blind"].loc[i]
        if b.correct_none:
            continue
        cor, none = b.correct, b.none_pos
        types = []
        if v.pred == none and b.pred == cor:
            types.append("A_audio_made_it_commit")
        if v.pred == cor and b.pred not in (cor, none):
            types.append("B_audio_misled")
        if a.correct_no_none == 1 and o.correct_no_none == 0 and v.correct_no_none == 0:
            types.append("C_audio_only_informative")
        if (v.pred not in (cor, none) and b.pred == v.pred
                and getattr(b, f"p_{b.pred}") > CONFIDENT):
            types.append("D_confident_wrong_video")
        for t in types:
            out.append({"type": t, "id": i, "site": b.site, "categories": b.correct_norm_categories,
                        "correct": cor, "correct_text": b.behaviors[cor - 1], "none_pos": none,
                        **{f"pred_{c}": by[c].loc[i].pred for c in CONDS},
                        **{f"p_correct_{c}": round(by[c].loc[i].p_correct, 3) for c in CONDS},
                        "options": " | ".join(f"{k + 1}. {t_}" for k, t_ in enumerate(b.behaviors))})
    ex = pd.DataFrame(out)
    ex.to_csv(OUT_DIR / f"{run}_failure_examples.csv", index=False)

    print("COUNTS BY TYPE")
    print(ex.type.value_counts().sort_index().to_string())
    for t in sorted(ex.type.unique()):
        sub = ex[ex.type == t].assign(_len=lambda d: d.options.str.len()).sort_values("_len").head(SHOW_PER_TYPE)
        print(f"\n=== {t} (shortest {SHOW_PER_TYPE}) ===")
        for x in sub.itertuples():
            print(f"{x.id}  [{x.site}; {x.categories}]")
            print(f"  {x.options}")
            print(f"  correct {x.correct}; predicted (options/video/audio/both) "
                  f"{x.pred_blind}/{x.pred_video}/{x.pred_audio}/{x.pred_both}; p(correct) "
                  f"{x.p_correct_blind}/{x.p_correct_video}/{x.p_correct_audio}/{x.p_correct_both}")
    print(f"\nWrote {run}_failure_examples.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()