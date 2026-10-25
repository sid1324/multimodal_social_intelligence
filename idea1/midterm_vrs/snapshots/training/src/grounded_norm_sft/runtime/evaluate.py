"""Native baseline evaluation artifacts (plan 9.A, 10.1, A4, A6).

The model sees input-only records (data_io whitelists); labels are joined only by the scorer.
Per example we save FP32 full-vocabulary log-probabilities at the task's answer token IDs, their
probabilities, the conditional distribution and log valid mass, the full-vocabulary argmax token
(the greedy next token), the answer IDs and prompt hash. Prediction rows follow the scorer schema
``normlens_pred_v1`` (metrics/normlens.py) plus provenance fields.

Failures are never dropped: example-level failures (missing image, image identity, quarantine,
serialization) become rows in failures.jsonl and the manifest is marked incomplete, which makes
the scorer refuse (missing predictions). Systematic failures stop the run: multi-token answers,
nonfinite logits, or MAX_CONSECUTIVE_FAILURES in a row. All files are write-once with SHA256
sidecars; the manifest records their hashes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..io_utils import write_once_json, write_once_jsonl
from ..objective import answer_logprobs, conditional_from_full
from ..serializer import MultiTokenAnswerError, QuarantineError, SerializationError
from .errors import ImageIdentityError, MissingImageError, NonfiniteError, RuntimeContractError

PRED_SCHEMA = "normlens_pred_v1"
EVAL_MANIFEST_SCHEMA = "gn_native_eval_v1"
MAX_CONSECUTIVE_FAILURES = 8
ROW_LEVEL_ERRORS = (MissingImageError, ImageIdentityError, QuarantineError, SerializationError)
STOP_ERRORS = (MultiTokenAnswerError, NonfiniteError)

# logits_fn(record) -> (logits_last [V], answer_ids, meta: dict)
LogitsFn = Callable[[Mapping], tuple]


def prediction_row(example_id: str, logits_last, answer_ids: Sequence[int], letters: Sequence[str], *, seed,
                   condition: str, task: str, arm: str, meta: Mapping | None = None) -> dict:
    import torch

    ids = [int(i) for i in answer_ids]
    if len(ids) != len(letters) or len(set(ids)) != len(ids):
        raise RuntimeContractError(f"{example_id}: answer ids {ids} do not match letters {list(letters)}")
    if logits_last.ndim != 1 or not all(0 <= i < logits_last.shape[0] for i in ids):
        raise RuntimeContractError(f"{example_id}: logits row/answer ids inconsistent with the vocabulary")
    if not bool(torch.isfinite(logits_last.float()).all()):  # checked before the shared objective's own guard
        raise NonfiniteError(f"{example_id}: nonfinite logits")
    full = answer_logprobs(logits_last, answer_ids)  # FP32 full-vocabulary log-probs at answer IDs
    if not bool(torch.isfinite(full).all()):
        raise NonfiniteError(f"{example_id}: nonfinite answer log-probabilities")
    cond, log_mass = conditional_from_full(full.double())
    logp_all = torch.log_softmax(logits_last.float(), dim=-1)
    top = int(torch.argmax(logp_all))
    row = {
        "example_id": example_id, "seed": seed, "condition": condition, "task": task, "arm": arm,
        "schema": PRED_SCHEMA,
        "full_logprobs": [float(x) for x in full.double()],
        "full_probs": [float(x) for x in full.double().exp()],
        "cond_logprobs": [float(x) for x in cond],
        "log_valid_mass": float(log_mass),
        "answer_letters": list(letters), "answer_token_ids": [int(i) for i in answer_ids],
        "argmax_token_id": top, "argmax_logprob": float(logp_all[top]),
        "argmax_is_answer": top in [int(i) for i in answer_ids],
    }
    if meta:
        row["serializer"] = dict(meta)
    return row


def run_native_eval(records: Iterable[Mapping], logits_fn: LogitsFn, letters: Sequence[str], *, seed, condition: str,
                    task: str, arm: str) -> tuple[list[dict], list[dict], dict]:
    rows, failures = [], []
    consecutive = 0
    t0 = time.monotonic()
    for rec in records:
        eid = rec["example_id"]
        try:
            logits, ids, meta = logits_fn(rec)
            rows.append(prediction_row(eid, logits, ids, letters, seed=seed, condition=condition, task=task, arm=arm,
                                       meta=meta))
            consecutive = 0
        except STOP_ERRORS:
            raise
        except ROW_LEVEL_ERRORS as exc:
            failures.append({"example_id": eid, "condition": condition, "task": task, "arm": arm,
                             "error_type": type(exc).__name__, "error": str(exc)})
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeContractError(f"{consecutive} consecutive evaluation failures (last: {exc}); "
                                           "systematic problem, stopping") from exc
    return rows, failures, {"seconds": time.monotonic() - t0}


def write_eval_artifacts(out_dir: str | Path, rows: list[dict], failures: list[dict], expected_ids: Sequence[str],
                         provenance: Mapping) -> dict:
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").exists():
        raise RuntimeContractError(f"{out_dir} already holds an evaluation manifest (immutable)")
    expected_ids = list(expected_ids)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise RuntimeContractError("expected_ids must be nonempty and unique (an empty evaluation is never complete)")
    got = [r["example_id"] for r in rows]
    failed = [f["example_id"] for f in failures]
    if set(got) & set(failed) or len(set(got)) != len(got) or len(set(failed)) != len(failed):
        raise RuntimeContractError("prediction/failure rows overlap or duplicate")
    accounted = set(got) | set(failed)
    missing = sorted(set(expected_ids) - accounted)
    extra = sorted(accounted - set(expected_ids))
    if extra:
        raise RuntimeContractError(f"rows for unexpected ids: {extra[:5]}")
    pred_sha = write_once_jsonl(out_dir / "predictions.jsonl", rows)
    fail_sha = write_once_jsonl(out_dir / "failures.jsonl", failures)
    manifest = {
        "schema": EVAL_MANIFEST_SCHEMA, "prediction_schema": PRED_SCHEMA,
        "expected": len(expected_ids), "predicted": len(rows), "failed": len(failures), "unaccounted": missing,
        "complete": not failures and not missing,
        "predictions_sha256": pred_sha, "failures_sha256": fail_sha,
        "expected_ids": expected_ids, "provenance": dict(provenance),
    }
    manifest["manifest_note"] = ("complete=false: the scorer must refuse these predictions (incomplete runs are "
                                 "never silently dropped)") if not manifest["complete"] else "complete"
    write_once_json(out_dir / "manifest.json", manifest)
    return manifest


# ------------------------------------------------------------------ resumable per-ID evaluation
RESUMABLE_MANIFEST_SCHEMA = "gn_native_eval_v2_resumable"


def _jsonl_bytes(rows: list[dict]) -> bytes:  # byte format of io_utils.write_once_jsonl
    import json

    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _json_bytes(obj) -> bytes:  # byte format of io_utils.write_once_json
    import json

    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_or_verify(path: Path, data: bytes) -> str:
    """Write once, or verify an existing artifact equals `data` exactly (completing only a missing
    sidecar for identical bytes, as a crash inside write_once_bytes can leave). Never replaces."""
    from ..hashing import sha256_bytes
    from ..io_utils import write_once_bytes

    digest = sha256_bytes(data)
    side = path.with_name(path.name + ".sha256")
    if not path.exists():
        if side.exists():
            raise RuntimeContractError(f"{side} exists without its artifact; refusing")
        return write_once_bytes(path, data)
    if path.read_bytes() != data:
        raise RuntimeContractError(f"existing {path.name} differs from the deterministic finalize output; "
                                   "refusing to replace (manual review)")
    if side.exists():
        if side.read_text().split()[0] != digest:
            raise RuntimeContractError(f"{side.name} does not match {path.name}")
    else:
        with open(side, "x") as fh:
            fh.write(digest + "  " + path.name + "\n")
    return digest


class ResumableNativeEval:
    """Per-ID write-once native evaluation for a fixed expected ID list.

    Each example's prediction row (prediction_row semantics) or row-level failure is persisted
    immediately as its own write-once file; a rerun of the same scientific identity skips finished
    IDs (never recomputed or replaced). The wall-clock budget is checked before every forward with
    a conservative reserve (the declared first-forward reserve, then the slowest measured example);
    on the cap the run returns `paused`. STOP_ERRORS and non-row errors propagate; row-level errors
    become failures (MAX_CONSECUTIVE_FAILURES per invocation stops). Finalization is deterministic
    and restart-safe (write or byte-verify; never replace); per-attempt provenance goes to
    eval_receipt_NNN.json, never into the manifest."""

    def __init__(self, out_dir: str | Path, identity: Mapping, records: Sequence[Mapping], logits_fn: LogitsFn,
                 letters: Sequence[str], *, seed, condition: str, task: str, arm: str, budget=None,
                 first_forward_reserve_s: float = 0.0):
        import math

        self.out = Path(out_dir)
        self.identity = dict(identity)
        self.records = list(records)
        self.ids = [r["example_id"] for r in self.records]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise RuntimeContractError("expected IDs must be nonempty and unique")
        if not (isinstance(first_forward_reserve_s, (int, float)) and math.isfinite(first_forward_reserve_s)
                and first_forward_reserve_s >= 0):
            raise RuntimeContractError("first_forward_reserve_s must be finite and nonnegative")
        self.logits_fn, self.letters = logits_fn, tuple(letters)
        self.fields = {"seed": seed, "condition": condition, "task": task, "arm": arm}
        self.budget, self.reserve = budget, float(first_forward_reserve_s)

    def _path(self, kind: str, i: str) -> Path:
        from ..hashing import sha256_text

        return self.out / kind / f"{sha256_text(i)[:32]}.json"

    def open(self) -> None:
        from ..io_utils import read_json

        if (self.out / "identity.json").exists():
            if read_json(self.out / "identity.json", verify=True) != self.identity:
                raise RuntimeContractError("existing evaluation directory belongs to a different run identity")
        else:
            write_once_json(self.out / "identity.json", self.identity)

    def state(self, i: str) -> str | None:
        item, fail = self._path("items", i).exists(), self._path("failures", i).exists()
        if item and fail:
            raise RuntimeContractError(f"{i}: both a prediction and a failure artifact exist; refusing to choose")
        return "item" if item else "failure" if fail else None

    def remaining(self) -> int:
        return sum(1 for i in self.ids if self.state(i) is None)

    def run(self) -> dict:
        consecutive = 0
        for rec in self.records:
            eid = rec["example_id"]
            if self.state(eid) is not None:
                continue
            if self.budget is not None and not self.budget.allows(self.reserve):
                return {"status": "paused", "reason": "time cap", "remaining": self.remaining()}
            t0 = time.monotonic()
            try:
                logits, ids, meta = self.logits_fn(rec)
                row = prediction_row(eid, logits, ids, self.letters, meta=meta, **self.fields)
                write_once_json(self._path("items", eid), row)
                consecutive = 0
            except STOP_ERRORS:
                raise
            except ROW_LEVEL_ERRORS as exc:
                write_once_json(self._path("failures", eid), {"example_id": eid, **self.fields,
                                                              "error_type": type(exc).__name__, "error": str(exc)})
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeContractError(f"{consecutive} consecutive evaluation failures (last: {exc}); "
                                               "systematic problem, stopping") from exc
            self.reserve = max(self.reserve, time.monotonic() - t0)
        return {"status": "items_complete", "remaining": 0}

    def finalize(self, provenance: Mapping) -> dict:
        from ..hashing import sha256_bytes, sha256_file
        from ..io_utils import read_json

        if self.remaining():
            raise RuntimeContractError(f"{self.remaining()} expected IDs unaccounted for; cannot finalize")
        rows, fails = [], []
        for i in self.ids:
            kind = self.state(i)
            r = read_json(self._path("items" if kind == "item" else "failures", i), verify=True)
            if r.get("example_id") != i or any(r.get(k) != v for k, v in self.fields.items()):
                raise RuntimeContractError(f"{kind} artifact for {i} does not belong to this run")
            if kind == "item" and r.get("schema") != PRED_SCHEMA:
                raise RuntimeContractError(f"prediction artifact for {i} has schema {r.get('schema')!r}")
            (rows if kind == "item" else fails).append(r)
        pred, fail = _jsonl_bytes(rows), _jsonl_bytes(fails)
        manifest = {
            "schema": RESUMABLE_MANIFEST_SCHEMA, "prediction_schema": PRED_SCHEMA, "identity": self.identity,
            "expected_ids": self.ids, "expected": len(self.ids), "predicted": len(rows), "failed": len(fails),
            "unaccounted": [], "complete": not fails, "predictions_sha256": sha256_bytes(pred),
            "failures_sha256": sha256_bytes(fail),
            "manifest_note": "complete" if not fails else ("complete=false: the scorer must refuse these predictions "
                                                           "(incomplete runs are never silently dropped)"),
        }
        _write_or_verify(self.out / "predictions.jsonl", pred)
        _write_or_verify(self.out / "failures.jsonl", fail)
        _write_or_verify(self.out / "manifest.json", _json_bytes(manifest))
        n = len(list(self.out.glob("eval_receipt_*.json")))
        write_once_json(self.out / f"eval_receipt_{n:03d}.json",
                        {"manifest_sha256": sha256_file(self.out / "manifest.json"), "provenance": dict(provenance)})
        return manifest


def dev_cond_nll(rows: list[dict], labels: Mapping[str, Mapping], ids: Sequence[str]) -> float:
    """Example-mean conditional vote NLL via the existing scorer (labels joined only here)."""
    from ..metrics.normlens import per_seed_values

    per_seed = per_seed_values(rows, labels, ids, condition="image")
    if len(per_seed) != 1:
        raise RuntimeContractError("dev selection expects exactly one seed")
    (vals,) = per_seed.values()
    return float(sum(v["cond_nll"] for v in vals.values()) / len(vals))
