"""
Score the human simulation: your answers vs. the EgoNormia key, by norm group and condition.

Reads (in p2_data_analysis/):
    human_sim_sheet_answers.csv   your filled-in sheet (falls back to human_sim_sheet.csv)
    human_sim_key.csv             groups + correct answers written by categorize_by_norm.py

Writes (in p2_data_analysis/):
    human_sim_scored.csv          one row per item: your answers, the key, right/wrong per condition
    human_sim_accuracy.png        accuracy by norm group, text-only vs. silent video

Usage:
    python score_human_sim.py
    python score_human_sim.py path/to/answers.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
KEY_CSV = SCRIPT_DIR / "human_sim_key.csv"
SCORED_CSV = SCRIPT_DIR / "human_sim_scored.csv"
FIG_PATH = SCRIPT_DIR / "human_sim_accuracy.png"
CHANCE = 0.25  # 4 real options + a blank "none of these" option on most items

GROUP_ORDER = ["Communication/Legibility", "Safety", "Proxemics"]
COLORS = {"text": "#2a78d6", "video": "#eb6834"}  # validated pair (CVD-safe)


def parse_answer(raw, row):
    """'option_3' -> 3; 'None' or '(blank)' -> index of the blank option; '' -> None."""
    s = str(raw).strip()
    if s in ("", "nan"):
        return None
    if s.lower().startswith("option_"):
        return int(s.split("_")[1])
    if s.isdigit():
        return int(s)
    if s.lower() in ("none", "(blank)", "blank"):
        for i in range(5):
            if str(row.get(f"option_{i}", "")).strip() == "(blank)":
                return i
    raise ValueError(f"Can't read answer {raw!r} for item {row['id']}")


def pct(x):
    return f"{x:.0%}" if pd.notna(x) else "  -"


def main():
    if len(sys.argv) > 1:
        answers_path = Path(sys.argv[1])
    else:
        answers_path = next((p for p in [SCRIPT_DIR / "human_sim_sheet_answers.csv",
                                         SCRIPT_DIR / "human_sim_sheet.csv"] if p.exists()), None)
    if answers_path is None or not KEY_CSV.exists():
        sys.exit("Need your answers sheet and human_sim_key.csv in p2_data_analysis/.")

    ans = pd.read_csv(answers_path, dtype=str, keep_default_na=False)  # keep "None" as text
    key = pd.read_csv(KEY_CSV, dtype={"sensibles": str}).fillna({"sensibles": ""})
    df = ans.merge(key[["id", "group", "correct", "sensibles", "video_source"]], on="id", how="left")
    if df["group"].isna().any():
        sys.exit("Some answer rows have no match in human_sim_key.csv. Were both made in the same run?")

    for cond in ("text", "video"):
        df[f"{cond}_idx"] = pd.array([parse_answer(r[f"{cond}_answer"], r) for _, r in df.iterrows()],
                                     dtype="Int64")
        df[f"{cond}_correct"] = [
            None if pd.isna(a) else int(int(a) == int(c)) for a, c in zip(df[f"{cond}_idx"], df["correct"])
        ]
        df[f"{cond}_sensible"] = [
            None if pd.isna(a) else int(str(int(a)) in s.split("|"))
            for a, s in zip(df[f"{cond}_idx"], df["sensibles"])
        ]

    def transition(t, v):
        if t is None or v is None or pd.isna(t) or pd.isna(v):
            return "unanswered"
        return {(1, 1): "right->right", (0, 1): "wrong->right",
                (1, 0): "right->wrong", (0, 0): "wrong->wrong"}[(t, v)]
    df["text_to_video"] = [transition(t, v) for t, v in zip(df["text_correct"], df["video_correct"])]

    keep = ["order", "id", "group", "video_source", "correct", "sensibles",
            "text_idx", "text_correct", "text_sensible", "video_idx", "video_correct",
            "video_sensible", "text_to_video", "speech_would_change_yes_maybe_no", "notes_cues_used"]
    df[[c for c in keep if c in df]].to_csv(SCORED_CSV, index=False)

    groups = [g for g in GROUP_ORDER if g in set(df["group"])]

    # ---- accuracy table ----
    print(f"Scored {len(df)} items from {answers_path.name}  (chance ~{CHANCE:.0%})\n")
    print(f"{'group':<26} {'n':>3} {'text acc':>9} {'video acc':>10} {'text sens':>10} {'video sens':>11}")
    rows = []
    for g in groups + ["ALL"]:
        sub = df if g == "ALL" else df[df["group"] == g]
        r = {
            "group": g, "n": len(sub),
            "text": pd.to_numeric(sub["text_correct"]).mean(),
            "video": pd.to_numeric(sub["video_correct"]).mean(),
            "text_s": pd.to_numeric(sub["text_sensible"]).mean(),
            "video_s": pd.to_numeric(sub["video_sensible"]).mean(),
        }
        rows.append(r)
        print(f"{g:<26} {r['n']:>3} {pct(r['text']):>9} {pct(r['video']):>10} "
              f"{pct(r['text_s']):>10} {pct(r['video_s']):>11}")
    print("  acc = picked the keyed correct answer; sens = picked any option the key marks sensible")

    # ---- what watching the video changed ----
    print("\nText-only -> silent video, by group:")
    order = ["right->right", "wrong->right", "right->wrong", "wrong->wrong", "unanswered"]
    trans = pd.crosstab(df["group"], df["text_to_video"]).reindex(index=groups, columns=order, fill_value=0)
    trans = trans.loc[:, (trans != 0).any()]
    print(trans.to_string())
    changed = (df["text_idx"] != df["video_idx"]).groupby(df["group"]).mean().reindex(groups)
    print("\nChanged answer after watching:", ", ".join(f"{g.split('/')[0]} {pct(v)}" for g, v in changed.items()))

    # ---- speech rating, if filled ----
    sp = df["speech_would_change_yes_maybe_no"].str.strip().str.lower()
    if (sp != "").any():
        print("\n'Would hearing speech change your answer?' by group:")
        print(pd.crosstab(df["group"], sp.replace("", "(blank)")).reindex(groups).fillna(0).astype(int).to_string())
    else:
        print("\n(speech_would_change column is empty, so no speech-rating summary)")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=200)
    width, gap = 0.36, 0.02
    xs = range(len(groups))
    by_group = {r["group"]: r for r in rows}
    for k, (cond, label) in enumerate([("text", "Text only"), ("video", "Silent video")]):
        offs = (-1 if k == 0 else 1) * (width / 2 + gap / 2)
        vals = [by_group[g][cond] * 100 for g in groups]
        bars = ax.bar([x + offs for x in xs], vals, width, color=COLORS[cond], label=label,
                      edgecolor="white", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                    va="bottom", fontsize=7.5, color="#444441")
    ax.axhline(CHANCE * 100, color="#8a8984", linestyle="--", linewidth=1)
    ax.text(len(groups) - 0.5, CHANCE * 100 + 1.5, "chance", ha="right", fontsize=7, color="#8a8984")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{g}\n(n={by_group[g]['n']})" for g in groups], fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=7, colors="#6b6a65")
    ax.yaxis.grid(True, color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_PATH)

    print(f"\nWrote {SCORED_CSV.name} and {FIG_PATH.name}")


if __name__ == "__main__":
    main()