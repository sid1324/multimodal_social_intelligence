"""NormLens scoring (plan 10.1, A4). Labels are joined here only.

Prediction rows (immutable JSONL written by the native evaluator; schema `normlens_pred_v1`):
  {"example_id": str, "seed": int|null, "condition": "image"|"text_only"|"shuffle",
   "full_logprobs": [lpA, lpB, lpC]}          # native: FP32 full-vocabulary log-probs
or {"example_id", "seed", "condition", "cond_logprobs": [...], "calibrated": true}  # B0-cal

Contracts:
  * primary: mean example-weighted conditional vote NLL over ALL examples incl. tied modes,
    original three-class vote distribution q;
  * full-vocabulary NLL and valid-answer mass reported for native rows (full = cond - log mass);
  * hard labels: unique human mode only (ties omitted); prediction ties -> smallest class ID;
  * macro-F1 over fixed labels [0,1,2], zero_division 0; Wrong/Okay subset (unique mode 0/1)
    keeps Impossible predictions and votes (no renormalization), subset F1 labels [0,1];
  * Brier = sum_c (p_c - q_c)^2 on conditional probabilities;
  * nonfinite predictions stop scoring; missing/extra/duplicate rows stop scoring;
  * metrics are computed per seed then averaged; probabilities are never averaged across seeds.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ..constants import ENTROPY_BINS, NORMLENS_CLASSES
from .bootstrap import cluster_bootstrap

KEY_SCALARS = ("cond_nll", "full_nll", "valid_mass", "brier", "accuracy", "macro_f1",
               "wo_cond_nll", "wo_accuracy", "wo_f1", "wo_pred_impossible_rate")


class ScoringError(RuntimeError):
    pass


class NonfiniteError(ScoringError):
    pass


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


FULL_MASS_LOG_TOL = 1e-6  # native label probabilities cannot sum above 1 beyond FP rounding


def validate_vote_q(q, where: str = "") -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (3,) or not np.all(np.isfinite(q)) or np.any(q < 0) or abs(q.sum() - 1) > 1e-9:
        raise ScoringError(f"{where}: vote distribution must be 3 finite nonnegative values summing to 1")
    return q


def validate_full_logprobs(lp, where: str = "") -> tuple[np.ndarray, float]:
    """Native full-vocabulary label log-probs: finite, and total label mass <= 1 (+tolerance)."""
    lp = np.asarray(lp, dtype=np.float64)
    if lp.shape != (3,) or not np.all(np.isfinite(lp)):
        raise NonfiniteError(f"{where}: nonfinite/invalid full log-probs; scoring stopped")
    log_mass = _logsumexp(lp)
    if log_mass > FULL_MASS_LOG_TOL:
        raise ScoringError(f"{where}: label probability mass exp({log_mass:.3g}) > 1; not native log-probs")
    return lp, log_mass


def per_example(row: Mapping, label: Mapping) -> dict:
    q = validate_vote_q(label["vote_probs"], str(row.get("example_id")))
    out: dict = {"example_id": row["example_id"]}
    if "full_logprobs" in row:
        lp, log_mass = validate_full_logprobs(row["full_logprobs"], str(row["example_id"]))
        cond_lp = lp - log_mass
        full_nll = float(-np.sum(q[q > 0] * lp[q > 0]))
        out.update(full_nll=full_nll, valid_mass=float(np.exp(log_mass)), log_valid_mass=log_mass)
    elif "cond_logprobs" in row:
        cond_lp = np.asarray(row["cond_logprobs"], dtype=np.float64)
        if cond_lp.shape != (3,) or not np.all(np.isfinite(cond_lp)):
            raise NonfiniteError(f"{row['example_id']}: nonfinite conditional log-probs; scoring stopped")
        if abs(_logsumexp(cond_lp)) > 1e-9:
            raise ScoringError(f"{row['example_id']}: conditional log-probs do not normalize")
    else:
        raise ScoringError("row has neither full_logprobs nor cond_logprobs")
    p = np.exp(cond_lp)
    cond_nll = float(-np.sum(q[q > 0] * cond_lp[q > 0]))
    if not np.isfinite(cond_nll):
        raise NonfiniteError(f"{row['example_id']}: nonfinite NLL; scoring stopped")
    if "full_nll" in out and abs(out["full_nll"] - (cond_nll - out["log_valid_mass"])) > 1e-9 * max(1.0, out["full_nll"]):
        raise ScoringError("full NLL != conditional NLL - log valid mass")
    out.update(cond_nll=cond_nll, brier=float(np.sum((p - q) ** 2)), p=p.tolist(),
               pred=int(np.argmax(cond_lp)),  # first maximum => smallest class ID on exact ties
               gold=label["unique_mode"])
    return out


def classification_report(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> dict:
    """Per-class precision/recall/F1 over fixed labels, zero_division=0; macro over all labels."""
    per = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        per[str(c)] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": tp + fn,
        }
    n = len(y_true)
    return {
        "n": n, "labels": list(labels),
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else None,
        "macro_f1": float(np.mean([per[str(c)]["f1"] for c in labels])) if n else None,
        "per_class": per,
        "missing_in_gold": [c for c in labels if c not in set(y_true)],
        "missing_in_pred": [c for c in labels if c not in set(y_pred)],
    }


def entropy_bin(h: float) -> str:
    for lo, hi in ENTROPY_BINS[:-1]:
        if lo <= h < hi:
            return f"[{lo:.4f},{hi:.4f})"
    lo, hi = ENTROPY_BINS[-1]
    if lo <= h <= hi:
        return f"[{lo:.4f},{hi:.4f}]"
    raise ScoringError(f"normalized entropy {h} outside [0,1]")


def summarize(per: list[dict], labels: Mapping[str, Mapping]) -> dict:
    if not per:
        return {"n": 0, "status": "N/A"}
    has_full = ["full_nll" in r for r in per]
    if any(has_full) and not all(has_full):
        raise ScoringError("mixed native/calibrated rows in one model")
    out: dict = {
        "n": len(per),
        "cond_nll": float(np.mean([r["cond_nll"] for r in per])),
        "brier": float(np.mean([r["brier"] for r in per])),
        "full_nll": float(np.mean([r["full_nll"] for r in per])) if all(has_full) else None,
        "valid_mass": float(np.mean([r["valid_mass"] for r in per])) if all(has_full) else None,
    }
    hard = [r for r in per if r["gold"] is not None]
    rep = classification_report([r["gold"] for r in hard], [r["pred"] for r in hard], NORMLENS_CLASSES)
    confusion = [[sum(1 for r in hard if r["gold"] == t and r["pred"] == p) for p in NORMLENS_CLASSES]
                 for t in NORMLENS_CLASSES]
    out.update(n_hard=len(hard), n_tied_omitted=len(per) - len(hard), accuracy=rep["accuracy"],
               macro_f1=rep["macro_f1"], per_class=rep["per_class"], missing_in_gold=rep["missing_in_gold"],
               missing_in_pred=rep["missing_in_pred"], confusion_gold_by_pred=confusion,
               impossible_precision=rep["per_class"]["2"]["precision"],
               impossible_recall=rep["per_class"]["2"]["recall"])
    wo = [r for r in hard if r["gold"] in (0, 1)]
    wo_rep = classification_report([r["gold"] for r in wo], [r["pred"] for r in wo], (0, 1))
    out.update(n_wo=len(wo),
               wo_cond_nll=float(np.mean([r["cond_nll"] for r in wo])) if wo else None,
               wo_accuracy=wo_rep["accuracy"], wo_f1=wo_rep["macro_f1"], wo_per_class=wo_rep["per_class"],
               wo_pred_impossible_rate=(sum(1 for r in wo if r["pred"] == 2) / len(wo)) if wo else None)
    strata: dict = {}
    for name, key in (("agreement", lambda r: labels[r["example_id"]]["agreement_stratum"]),
                      ("entropy", lambda r: entropy_bin(labels[r["example_id"]]["normalized_entropy"]))):
        groups: dict = defaultdict(list)
        for r in per:
            groups[key(r)].append(r)
        strata[name] = {k: {"n": len(v), "cond_nll": float(np.mean([r["cond_nll"] for r in v])),
                            "accuracy_hard": (lambda h: sum(r["pred"] == r["gold"] for r in h) / len(h) if h else None)(
                                [r for r in v if r["gold"] is not None])}
                        for k, v in sorted(groups.items())}
    out["strata"] = strata
    return out


def _by_seed(rows: Iterable[Mapping], expected_ids: Sequence[str], condition: str) -> dict:
    if len(set(expected_ids)) != len(expected_ids):
        raise ScoringError("duplicate canonical example IDs in the expected set")
    seeds: dict = defaultdict(dict)
    for r in rows:
        if r.get("condition", "image") != condition:
            continue
        bucket = seeds[r.get("seed")]
        if r["example_id"] in bucket:
            raise ScoringError(f"duplicate prediction for {r['example_id']} seed {r.get('seed')}")
        bucket[r["example_id"]] = r
    if not seeds:
        raise ScoringError(f"no rows for condition {condition!r}")
    want = set(expected_ids)
    for s, bucket in seeds.items():
        if set(bucket) != want:
            raise ScoringError(f"seed {s}: {len(want - set(bucket))} missing, {len(set(bucket) - want)} extra "
                               "predictions (incomplete runs are never silently dropped)")
    return dict(seeds)


def per_seed_values(rows, labels: Mapping[str, Mapping], ids: Sequence[str], condition: str = "image") -> dict:
    """{seed: {example_id: per_example dict}}"""
    return {s: {i: per_example(b[i], labels[i]) for i in ids}
            for s, b in _by_seed(rows, ids, condition).items()}


def score_model(rows, labels: Mapping[str, Mapping], ids: Sequence[str], condition: str = "image") -> dict:
    per_seed = per_seed_values(rows, labels, ids, condition)
    summaries = {str(s): summarize(list(v.values()), labels) for s, v in per_seed.items()}
    seed_mean, seed_sd = {}, {}
    for k in KEY_SCALARS:
        vals = [summ[k] for summ in summaries.values() if summ.get(k) is not None]
        if len(vals) == len(summaries) and vals:
            seed_mean[k] = float(np.mean(vals))
            seed_sd[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return {"condition": condition, "n_seeds": len(summaries), "per_seed": summaries,
            "seed_mean": seed_mean, "seed_sd": seed_sd,
            "label": "single-seed pilot" if len(summaries) == 1 else f"{len(summaries)}-seed"}


FROZEN_COMPARATORS = ("B0", "B0-cal")


def contrast(rows_a, rows_b, labels: Mapping[str, Mapping], ids: Sequence[str], metric: str = "cond_nll",
             subset: Callable[[Mapping], bool] | None = None, condition: str = "image",
             b_frozen: bool = False) -> dict:
    """Paired A-B on example means of seed-averaged per-example values; group bootstrap.

    Learned arms must have identical seed IDs. A single seed-less comparator is broadcast across A's
    seeds only when b_frozen=True (the frozen B0/B0-cal baselines)."""
    sel = [i for i in ids if subset is None or subset(labels[i])]
    a = per_seed_values(rows_a, labels, ids, condition)
    b = per_seed_values(rows_b, labels, ids, condition)
    for name, d in (("A", a), ("B", b)):
        if any(metric not in v[i] for v in d.values() for i in sel):
            raise ScoringError(f"metric {metric} unavailable for model {name}")
    a_bar = {i: float(np.mean([a[s][i][metric] for s in a])) for i in sel}
    b_bar = {i: float(np.mean([b[s][i][metric] for s in b])) for i in sel}
    diffs = [a_bar[i] - b_bar[i] for i in sel]
    boot = cluster_bootstrap(diffs, [labels[i]["group_id"] for i in sel])
    sa, sb = sorted(a, key=str), sorted(b, key=str)
    if b_frozen:
        if sb != [None]:
            raise ScoringError("frozen comparator must have exactly one seed-less (seed=null) run")
        if None in sa:
            raise ScoringError("learned arm rows must carry actual seed IDs")
        pairs = [(s, None) for s in sa]
    elif sa == sb and None not in sa:
        pairs = [(s, s) for s in sa]
    else:
        raise ScoringError(f"seed IDs do not match ({sa} vs {sb}); broadcasting is only allowed for a "
                           "frozen B0/B0-cal comparator")
    per_seed = {f"{x}-{y}": float(np.mean([a[x][i][metric] - b[y][i][metric] for i in sel])) if sel else None
                for x, y in pairs}
    vals = [v for v in per_seed.values() if v is not None]
    return {"metric": metric, "n": len(sel), "point": float(np.mean(diffs)) if diffs else None,
            "bootstrap": boot, "per_seed_differences": per_seed,
            "seed_diff_mean": float(np.mean(vals)) if vals else None,
            "seed_diff_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "favorable_direction": "negative", "label": "single-seed pilot" if len(pairs) == 1 else "multi-seed"}


def wrong_okay_subset(label: Mapping) -> bool:
    return label["unique_mode"] in (0, 1)


def image_dependence(rows_correct, rows_shuffled, labels, ids) -> dict:
    """D = NLL(shuffled image) - NLL(correct image) on the Wrong/Okay subset with ORIGINAL targets,
    plus change in P(Impossible). Descriptive dependence diagnostic, not a counterfactual label."""
    sel = [i for i in ids if wrong_okay_subset(labels[i])]
    c = per_seed_values(rows_correct, labels, ids, "image")
    s = per_seed_values(rows_shuffled, labels, ids, "shuffle")
    if sorted(c, key=str) != sorted(s, key=str):
        raise ScoringError("seed sets differ between correct and shuffled conditions")
    d = [float(np.mean([s[k][i]["cond_nll"] - c[k][i]["cond_nll"] for k in c])) for i in sel]
    dp = [float(np.mean([s[k][i]["p"][2] - c[k][i]["p"][2] for k in c])) for i in sel]
    return {"n_wo": len(sel), "D_mean": float(np.mean(d)) if d else None,
            "delta_p_impossible_mean": float(np.mean(dp)) if dp else None}


def context_strata(examples: list[dict]) -> dict:
    """Descriptive natural context/prior-breaking counts per held-out split (A4).
    Ties and Impossible modal categories never qualify."""
    train = defaultdict(list)
    for e in examples:
        if e["split"] == "train":
            train[e["action_canonical"]].append(e)
    out = {}
    for split in ("dev", "test"):
        rows = [e for e in examples if e["split"] == split and e["unique_mode"] in (0, 1)]
        by_action, by_image = defaultdict(list), defaultdict(list)
        for e in rows:
            by_action[e["action_canonical"]].append(e)
            by_image[e["rgb_sha256"]].append(e)
        rep_actions = [a for a, v in by_action.items()
                       if len({e["rgb_sha256"] for e in v}) > 1 and {0, 1} <= {e["unique_mode"] for e in v}]
        same_image = [g for g, v in by_image.items()
                      if len({e["action_canonical"] for e in v}) > 1 and {0, 1} <= {e["unique_mode"] for e in v}]
        prior_breaking, train_disagree = [], []
        for e in rows:
            tr = train.get(e["action_canonical"])
            if not tr:
                continue
            mean_q = np.mean([t["vote_probs"] for t in tr], axis=0)
            top = np.flatnonzero(mean_q == mean_q.max())
            if len(top) == 1 and int(top[0]) in (0, 1) and int(top[0]) != e["unique_mode"]:
                prior_breaking.append(e["example_id"])
                if len({t["unique_mode"] for t in tr}) > 1:
                    train_disagree.append(e["example_id"])
        out[split] = {
            "repeated_action_opposite_wo_actions": len(rep_actions),
            "repeated_action_opposite_wo_examples": sum(len(by_action[a]) for a in rep_actions),
            "same_image_different_action_wo_groups": len(same_image),
            "same_image_different_action_wo_examples": sum(len(by_image[g]) for g in same_image),
            "prior_breaking_examples": sorted(prior_breaking),
            "prior_breaking_with_training_disagreement": sorted(train_disagree),
        }
    return out
