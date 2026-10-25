"""Generate coverage-aware tables and clustered uncertainty from raw predictions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from social_rl.common import ROOT, read_jsonl, write_json
from social_rl.tasks import item_metrics


def cluster_ci(rows, values, iterations=2000, seed=777):
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[row["group"]].append(value)
    arrays = [np.asarray(x, dtype=float) for x in groups.values()]
    if len(arrays) < 2:
        return [None, None]
    sums = np.asarray([a.sum() for a in arrays])
    counts = np.asarray([len(a) for a in arrays])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arrays), size=(iterations, len(arrays)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    all_rows = read_jsonl(args.manifest)
    if any(r.get('split') == 'final' for r in all_rows):
        raise ValueError("Final partition is reserved")
    by_id = {r["id"]: r for r in all_rows}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    conditions = {}
    for directory in args.runs:
        path = Path(directory)
        for pred in read_jsonl(path / "predictions.jsonl"):
            if pred["id"] not in by_id:
                raise ValueError("Prediction ID outside evaluation manifest")
            key = path.name + ":" + pred["condition"]
            conditions.setdefault(key, {})[pred["id"]] = pred
    summaries, categories, official, item_rows = {}, {}, {}, []
    for condition, predictions in conditions.items():
        ids = sorted(predictions)
        rows = [by_id[i] for i in ids]
        metrics = [item_metrics(r, predictions[r["id"]]) for r in rows]
        summary = {"n_attempted": len(ids), "n_panel": len(all_rows), "source_groups": len({r['group'] for r in rows}),
                   "n_with_inference_errors": sum(bool(p['errors']) for p in predictions.values()),
                   "n_with_invalid_output": sum(not all(p['valid'].values()) for p in predictions.values()),
                   "total_inference_seconds": sum(p['seconds'] for p in predictions.values())}
        for metric in metrics[0]:
            values = [m[metric] for m in metrics]
            summary[metric] = float(np.mean(values))
            summary[metric + "_ci95"] = cluster_ci(rows, values)
        summaries[condition] = summary
        categories[condition] = {}
        for row, metric in zip(rows, metrics):
            pred = predictions[row['id']]
            item_rows.append({"condition": condition, "id": row['id'], "group": row['group'], **metric})
            if row['dataset'] == 'ego':
                names = row["taxonomy"].get(str(row["answer"])) or []
                for name in names:
                    categories[condition].setdefault(name, []).append(metric)
                official.setdefault(row['id'], {})[condition] = {
                    "best": {"results": [pred.get('action', -1), pred.get('justification', -1)],
                             "correct": [row['answer'], row['answer']]},
                    "sensible": {"results": pred.get('sensible', []), "correct": row['sensible']}}
        categories[condition] = {name: {"n": len(ms), **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}}
                                  for name, ms in categories[condition].items()}
        verified_ids = [i for i in ids if by_id[i].get('verified')]
        if verified_ids:
            summaries[condition]['verified'] = {"n": len(verified_ids), **{m: float(np.mean([
                item_metrics(by_id[i], predictions[i])[m] for i in verified_ids])) for m in metrics[0]}}
    comparisons = {}
    names = list(conditions)
    if len(names) > 1:
        ref = names[0]
        for name in names[1:]:
            common_ids = sorted(set(conditions[ref]) & set(conditions[name]))
            if not common_ids:
                continue
            delta = {}
            rows = [by_id[i] for i in common_ids]
            for metric in item_metrics(rows[0], conditions[ref][rows[0]['id']]):
                values = [item_metrics(by_id[i], conditions[name][i])[metric] - item_metrics(by_id[i], conditions[ref][i])[metric]
                          for i in common_ids]
                delta[metric] = {"difference": float(np.mean(values)), "ci95": cluster_ci(rows, values)}
            comparisons[name + " minus " + ref] = {"n_common": len(common_ids), **delta}
    write_json(out / "summary.json", {"conditions": summaries, "category_membership": "correct-action taxonomy, overlapping",
                                     "categories": categories, "paired_common_panel": comparisons})
    write_json(out / "official_eval.json", official)
    write_json(out / "official_split.json", {"split": sorted(official)})
    if item_rows:
        with (out / 'per_example.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(item_rows[0]))
            writer.writeheader()
            writer.writerows(item_rows)
    lines = ["# Measured results", "", "All numbers below are computed from saved predictions. Failures remain in denominators.", ""]
    if all_rows[0]['dataset'] == 'r3':
        lines += ["| Condition | Attempted/panel | Accuracy | Invalid examples |",
                  "|---|---:|---:|---:|"]
        for name, s in summaries.items():
            lines.append(f"| {name} | {s['n_attempted']}/{s['n_panel']} | {100*s['accuracy']:.1f} | {s['n_with_invalid_output']} |")
    else:
        lines += ["| Condition | Attempted/panel | Action | Justification | Joint | Sensibility IoU | Invalid examples |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, s in summaries.items():
            vals = [f"{100*s[k]:.1f}" if k in s else '—' for k in ['action', 'justification', 'joint', 'sensibility_iou']]
            lines.append(f"| {name} | {s['n_attempted']}/{s['n_panel']} | " + " | ".join(vals) + f" | {s['n_with_invalid_output']} |")
    lines += ["", "Scores are percentages. Per-condition tables may have different coverage while jobs run;",
              "use paired_common_panel in summary.json for matched comparisons. Cluster-bootstrap intervals",
              "describe example sampling uncertainty, not variability across training seeds."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__':
    main()
