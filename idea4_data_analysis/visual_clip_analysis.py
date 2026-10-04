"""
Visual modality analysis: are norm categories separable from what's visible?

For each audio-covered EgoNormia item in the three exclusive groups
(Communication/Legibility, Safety, Proxemics; tagged with one and neither of the others),
sample frames from video_prev.mp4, embed them with CLIP, and average into one vector per item.
Then:
  (1) linear probe: can a logistic regression on the CLIP vector predict the norm group?
      (5-fold cross-validated accuracy vs. majority baseline, plus per-group recall)
  (2) confound check: the same probe predicting collection site instead
  (3) t-SNE plot, colored by norm group and by site

Reads:
    p2_data_analysis/egonormia_audio_by_norm.csv   (from categorize_by_norm.py)
    your EgoNormia clip folders (found automatically, or pass the path)

Writes (in p2_data_analysis/):
    clip_embeddings.npz          cached embeddings, so reruns skip the slow step
    visual_probe_predictions.csv per-item probe prediction (for picking qualitative examples)
    visual_tsne.png              two-panel t-SNE figure

Setup (once):
    pip install torch transformers opencv-python scikit-learn matplotlib pandas
Usage:
    python visual_clip_analysis.py
    python visual_clip_analysis.py /path/to/clips_root
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Recent NumPy on macOS (Accelerate) prints spurious "divide by zero / overflow in matmul"
# warnings from scikit-learn; results are unaffected, so hide them.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

# ---- settings ----
MODEL_NAME = "openai/clip-vit-base-patch32"
FRAMES_PER_CLIP = 5        # video_prev is ~5 s, so about one frame per second
MAX_PER_GROUP = None       # e.g. 150 to run faster; None = use every item in each pool
SEED = 0
WITHIN_SITES = ["iiith", "frl_track_1_public", "minnesota"]  # norm probe run inside each site alone
MIN_PER_GROUP_IN_SITE = 10   # skip a site if any group has fewer items than this there

SCRIPT_DIR = Path(__file__).resolve().parent
BY_NORM_CSV = SCRIPT_DIR / "egonormia_audio_by_norm.csv"
EMB_CACHE = SCRIPT_DIR / "clip_embeddings.npz"
PRED_CSV = SCRIPT_DIR / "visual_probe_predictions.csv"
FIG_PATH = SCRIPT_DIR / "visual_tsne.png"

GROUPS = {
    "Communication/Legibility": "cat_communication_legibility",
    "Safety": "cat_safety",
    "Proxemics": "cat_proxemics",
}
GROUP_STYLE = {  # validated categorical slots 1-3, plus marker shape as a second cue
    "Communication/Legibility": ("#2a78d6", "o"),
    "Safety": ("#eb6834", "s"),
    "Proxemics": ("#1baf7a", "^"),
}
SITE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
OTHER_COLOR = "#a8a79f"


def find_clips_root(ids):
    ids = set(ids)
    for dirpath, dirnames, _ in os.walk(SCRIPT_DIR.parent):
        if any(d in ids for d in dirnames):
            return Path(dirpath)
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "human_sim_clips"]
    sys.exit("Couldn't find the clip folders. Pass the path: python visual_clip_analysis.py /path/to/clips_root")


def load_items():
    if not BY_NORM_CSV.exists():
        sys.exit(f"{BY_NORM_CSV.name} not found. Run categorize_by_norm.py first.")
    df = pd.read_csv(BY_NORM_CSV)
    parts = []
    for g, col in GROUPS.items():
        others = [c for c in GROUPS.values() if c != col]
        pool = df[(df[col] == 1) & (df[others] == 0).all(axis=1)].copy()
        if MAX_PER_GROUP and len(pool) > MAX_PER_GROUP:
            pool = pool.sample(MAX_PER_GROUP, random_state=SEED)
        pool["group"] = g
        parts.append(pool)
    return pd.concat(parts, ignore_index=True)


def sample_frames(video_path, n):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for idx in np.linspace(0, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def embed_items(items, clips_root):
    if EMB_CACHE.exists():
        cache = np.load(EMB_CACHE, allow_pickle=True)
        cached = dict(zip(cache["ids"], cache["emb"]))
    else:
        cached = {}
    todo = [i for i in items["id"] if i not in cached]

    if todo:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Embedding {len(todo)} clips with {MODEL_NAME} on {device} (cached: {len(cached)})")
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        proc = CLIPProcessor.from_pretrained(MODEL_NAME)

        for k, item_id in enumerate(todo, 1):
            frames = sample_frames(clips_root / item_id / "video_prev.mp4", FRAMES_PER_CLIP)
            if not frames:  # fall back to the provided frame strip
                strip = clips_root / item_id / "frame_all_prev.jpg"
                if strip.exists():
                    frames = [np.array(Image.open(strip).convert("RGB"))]
            if not frames:
                print(f"  skipped (no video or frames): {item_id}")
                continue
            with torch.no_grad():
                inputs = proc(images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(device)
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vec = feats.mean(0)
                cached[item_id] = (vec / vec.norm()).cpu().numpy()
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}")
                np.savez(EMB_CACHE, ids=np.array(list(cached)), emb=np.stack(list(cached.values())))

    have = items["id"].isin(cached)
    items = items[have].reset_index(drop=True)
    X = np.stack([cached[i] for i in items["id"]])
    return items, X


def probe(X, y, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    pred = cross_val_predict(clf, X, y, cv=cv)
    classes = sorted(set(y))
    chance = 1 / len(classes)
    print(f"\n{label}")
    print(f"  balanced accuracy (5-fold): {scores.mean():.1%} ± {scores.std():.1%}   (chance {chance:.1%})")
    rec = recall_score(y, pred, labels=classes, average=None)
    for c, r in zip(classes, rec):
        print(f"    recall  {c:<26} {r:.1%}   (n={int((np.array(y) == c).sum())})")
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
    print("  confusion matrix (rows = true, cols = predicted):")
    print("  " + cm.to_string().replace("\n", "\n  "))
    return pred


def plot(items, X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=200)

    ax = axes[0]
    for g, (color, marker) in GROUP_STYLE.items():
        m = items["group"] == g
        ax.scatter(Z[m, 0], Z[m, 1], s=14, c=color, marker=marker, edgecolors="white",
                   linewidths=0.4, label=f"{g} (n={m.sum()})", alpha=0.9)
    ax.set_title("Colored by norm group", fontsize=9, color="#2b2b29")

    ax = axes[1]
    top_sites = items["video_source"].value_counts().index[:len(SITE_COLORS)]
    for site, color in zip(top_sites, SITE_COLORS):
        m = items["video_source"] == site
        ax.scatter(Z[m, 0], Z[m, 1], s=14, c=color, edgecolors="white", linewidths=0.4,
                   label=f"{site} (n={m.sum()})", alpha=0.9)
    m = ~items["video_source"].isin(top_sites)
    if m.any():
        ax.scatter(Z[m, 0], Z[m, 1], s=14, c=OTHER_COLOR, edgecolors="white", linewidths=0.4,
                   label=f"other (n={m.sum()})", alpha=0.9)
    ax.set_title("Colored by collection site", fontsize=9, color="#2b2b29")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e6e5df")
        ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.suptitle("t-SNE of CLIP embeddings of video_prev frames", fontsize=9.5, color="#2b2b29")
    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_inches="tight")


def main():
    items = load_items()
    clips_root = Path(sys.argv[1]) if len(sys.argv) > 1 else find_clips_root(items["id"])
    print(f"Clips folder: {clips_root}")
    print("Items per group:", items["group"].value_counts().to_dict())

    items, X = embed_items(items, clips_root)
    print(f"Embedded items: {len(items)}")

    pred = probe(X, items["group"].tolist(), "(1) NORM GROUP PROBE: CLIP frames -> norm group")
    items["probe_pred"] = pred

    site_counts = items["video_source"].value_counts()
    keep_sites = site_counts[site_counts >= 10].index
    m = items["video_source"].isin(keep_sites).to_numpy()
    probe(X[m], items.loc[m, "video_source"].tolist(),
          f"(2) SITE PROBE (confound check): CLIP frames -> site ({len(keep_sites)} sites with >=10 items)")

    print("\n(3) WITHIN-SITE NORM PROBE (site held constant, so it can't drive the prediction)")
    for site in WITHIN_SITES:
        ms = (items["video_source"] == site).to_numpy()
        counts = items.loc[ms, "group"].value_counts()
        print(f"\n  site = {site}: " + ", ".join(f"{g.split('/')[0]} {counts.get(g, 0)}" for g in GROUPS))
        if len(counts) < len(GROUPS) or counts.min() < MIN_PER_GROUP_IN_SITE:
            print(f"    skipped: a group has fewer than {MIN_PER_GROUP_IN_SITE} items here")
            continue
        probe(X[ms], items.loc[ms, "group"].tolist(), f"  norm probe within {site}")

    items[["id", "group", "video_source", "probe_pred"]].to_csv(PRED_CSV, index=False)
    plot(items, X)
    print(f"\nWrote {PRED_CSV.name} and {FIG_PATH.name}")


if __name__ == "__main__":
    main()