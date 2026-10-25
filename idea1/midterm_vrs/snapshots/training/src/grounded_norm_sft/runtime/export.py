"""Native export verification (plan 9.D.13).

Contract: save ordinary unmerged PEFT adapters (FP32, never downcast) plus processor/config; then
two clean fresh standard HF+PEFT reloads, running only pinned native source with no project
hooks or patched forwards, must reproduce the in-memory pre-save native inference on a fixed,
nonempty fixture set: identical argmax tokens and answer FP32 log-probability
|diff| <= max(1e-6, 10 x native_repeat_floor). If the native repeat floor exceeds 1e-4, stop for
a numerical/backend decision instead of widening the tolerance. Every fixture output must be
present, finite and of the declared answer length; empty or nonfinite inputs fail closed.
Reloaded adapter tensors must be bitwise equal to the saved FP32 tensors (nonempty).
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from ..constants import EXPORT_FLOOR_MULTIPLIER, EXPORT_FLOOR_STOP, EXPORT_LOGPROB_ABS_MIN
from .errors import ExportError

PROJECT_PACKAGE = "grounded_norm_sft"
NATIVE_MODELING_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"

# fixture_fn(model) -> {example_id: {"full_logprobs": [...], "argmax_token_id": int}}
FixtureFn = Callable[[object], Mapping[str, Mapping]]


def validate_fixture_outputs(out: Mapping[str, Mapping], fixture_ids: Sequence[str], n_answers: int) -> None:
    if not fixture_ids or len(set(fixture_ids)) != len(fixture_ids):
        raise ExportError("fixture ids must be a nonempty unique list")
    if set(out) != set(fixture_ids):
        raise ExportError(f"fixture outputs cover {sorted(out)[:4]}, expected exactly the fixed fixture ids")
    for k in fixture_ids:
        lp = out[k].get("full_logprobs")
        if not isinstance(lp, (list, tuple)) or len(lp) != n_answers:
            raise ExportError(f"{k}: answer log-prob vector must have length {n_answers}")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x <= 0 for x in lp):
            raise ExportError(f"{k}: nonfinite or invalid answer log-probs {lp}")
        a = out[k].get("argmax_token_id")
        if not isinstance(a, int) or isinstance(a, bool) or a < 0:
            raise ExportError(f"{k}: invalid argmax token id {a!r}")


def max_abs_diff(a: Mapping[str, Mapping], b: Mapping[str, Mapping], fixture_ids: Sequence[str], n_answers: int) -> float:
    validate_fixture_outputs(a, fixture_ids, n_answers)
    validate_fixture_outputs(b, fixture_ids, n_answers)
    return max(abs(float(x) - float(y)) for k in fixture_ids
               for x, y in zip(a[k]["full_logprobs"], b[k]["full_logprobs"]))


def native_repeat_floor(fixture_fn: FixtureFn, model, repeats: int, fixture_ids: Sequence[str],
                        n_answers: int) -> tuple[float, list]:
    """Max |diff| of answer log-probs across repeated untouched native forwards."""
    if repeats < 2:
        raise ExportError("repeat floor needs at least two repeats")
    runs = [fixture_fn(model) for _ in range(repeats)]
    floor = max(max_abs_diff(runs[0], r, fixture_ids, n_answers) for r in runs[1:])
    if any(runs[0][k]["argmax_token_id"] != r[k]["argmax_token_id"] for r in runs[1:] for k in fixture_ids):
        raise ExportError("native repeated forwards change the argmax token; numerical/backend decision needed")
    return floor, runs


def export_tolerance(floor: float) -> float:
    if not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 0:
        raise ExportError(f"invalid native repeat floor {floor!r}")
    if floor > EXPORT_FLOOR_STOP:
        raise ExportError(f"native repeat floor {floor:.3g} > {EXPORT_FLOOR_STOP}: stop for a numerical/backend "
                          "decision (tolerance is never widened automatically)")
    return max(EXPORT_LOGPROB_ABS_MIN, EXPORT_FLOOR_MULTIPLIER * floor)


def compare_to_reference(ref: Mapping[str, Mapping], other: Mapping[str, Mapping], tol: float,
                         fixture_ids: Sequence[str], n_answers: int) -> dict:
    if not math.isfinite(tol) or tol <= 0:
        raise ExportError("tolerance must be finite and positive")
    diff = max_abs_diff(ref, other, fixture_ids, n_answers)
    argmax_mismatch = sorted(k for k in fixture_ids if ref[k]["argmax_token_id"] != other[k]["argmax_token_id"])
    return {"max_abs_logprob_diff": diff, "tolerance": tol, "argmax_mismatch": argmax_mismatch,
            "passed": diff <= tol and not argmax_mismatch}


def hook_inventory(model) -> list[dict]:
    """Every registered forward/backward hook with the module that defined its function."""
    out = []
    for name, mod in model.named_modules():
        for kind in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
            for fn in getattr(mod, kind, {}).values():
                target = getattr(fn, "__func__", fn)
                out.append({"module": name, "kind": kind, "fn_module": getattr(target, "__module__", None),
                            "fn": getattr(target, "__qualname__", repr(target))})
    return out


def assert_no_project_hooks(model) -> list[dict]:
    inv = hook_inventory(model)
    ours = [h for h in inv if str(h["fn_module"] or "").startswith(PROJECT_PACKAGE)]
    if ours:
        raise ExportError(f"project hooks present on the native model: {ours[:3]}")
    return inv


def assert_native_source(model) -> dict:
    """No project classes, no instance-level forward patches, Qwen3.5 classes from pinned source."""
    classes: dict[str, int] = {}
    for name, mod in model.named_modules():
        cls = type(mod)
        if cls.__module__.startswith(PROJECT_PACKAGE):
            raise ExportError(f"{name}: project module class {cls.__qualname__} in the native model")
        if "forward" in vars(mod):
            raise ExportError(f"{name}: instance-level forward override (monkeypatch) present")
        if cls.__name__.startswith("Qwen3_5") and cls.__module__ != NATIVE_MODELING_MODULE:
            raise ExportError(f"{name}: {cls.__name__} defined in {cls.__module__}, not the pinned modeling module")
        key = f"{cls.__module__}.{cls.__qualname__}"
        classes[key] = classes.get(key, 0) + 1
    return {"module_classes": classes, "hooks": assert_no_project_hooks(model)}


def assert_adapter_equal(saved: Mapping, reloaded: Mapping) -> dict:
    """Bitwise FP32 equality of saved vs reloaded adapter tensors (names normalized by caller)."""
    import torch

    if not saved or not reloaded:
        raise ExportError("adapter tensor mapping is empty")
    if set(saved) != set(reloaded):
        raise ExportError(f"adapter key sets differ: {sorted(set(saved) ^ set(reloaded))[:4]}")
    for k in saved:
        a, b = saved[k], reloaded[k].detach().cpu()
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ExportError(f"{k}: adapter dtype {a.dtype}/{b.dtype}; FP32 required (no export downcast)")
        if a.shape != b.shape or not torch.equal(a, b):
            raise ExportError(f"{k}: reloaded adapter tensor differs from the saved tensor")
    return {"tensors": len(saved), "bitwise_equal": True}


def verify_reloads(reference: Mapping[str, Mapping], floor: float, reload_fns: Sequence[Callable[[], Mapping]],
                   fixture_ids: Sequence[str], n_answers: int) -> dict:
    if len(reload_fns) < 2:
        raise ExportError("two independent fresh reloads are required")
    validate_fixture_outputs(reference, fixture_ids, n_answers)
    tol = export_tolerance(floor)
    results = [compare_to_reference(reference, fn(), tol, fixture_ids, n_answers) for fn in reload_fns]
    report = {"native_repeat_floor": floor, "tolerance": tol, "reloads": results, "fixtures": list(fixture_ids),
              "passed": all(r["passed"] for r in results)}
    if not report["passed"]:
        raise ExportError(f"fresh reload differs from pre-save native inference: {results}")
    return report
