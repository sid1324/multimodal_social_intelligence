"""Native EgoNormia transfer generation (plan A5, A5.1, Appendix B). Executable runtime; nothing
here has been run against real weights.

Inputs: only `id`, the five `behaviors` and five `justifications` are kept from the validated
production source (gold indices are dropped in `inputs_from_items`, before any prompt exists), plus
the precomputed pre-action 1024x512 frame sheet (media session's cache, SHA256-checked).

Per item, independent native calls (fresh cache, rope_deltas reset, model.eval, no hooks):
  best        visual Appendix B1 prompt
  justification  visual B2 prompt with the MODEL-selected behavior only; skipped (null) when the
              strict best-action parse is invalid
  sensible    visual B3 prompt (primary endpoint, original order)
  sensible_text_only  B3 text-only prompt, explicit text_only=True, no image
Generation: model.generate(do_sample=False, max_new_tokens=256, use_cache=True) with the model's
NATIVE generation_config otherwise untouched. Its EOS id(s) (pinned config: 248044) and the chat
template's assistant-turn terminator <|im_end|> (248046) are distinct and both recorded; neither is
changed or guessed. `raw` is the decoded assistant turn up to the first native EOS or terminator;
`raw_full`, token IDs, EOS/terminator positions, length and cap flags are kept as evidence.

Diagnostics (cohort diag32, paired with a completed full400 run; primary generation is not
re-run): four cyclic rotations of the sensible prompt (generation), and 32 legal-answer candidate
full-sequence log-probabilities (answer tokens + native assistant terminator, summed, never length
normalized) on the original-order sensible prompt, with prompt-prefix and prefix-free checks.

Artifacts: per-ID write-once item/failure files (resume after a time cap never redoes or replaces
an ID), then a write-once predictions.jsonl (scorer schema ego_pred_v1) + manifest once every
cohort ID is accounted for. Decode failures from the sheet manifest are the common exclusion.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..constants import (EGO_CANDIDATE_MASS_TOL, EGO_EXPECTED_VISUAL_TOKENS, EGO_MAX_NEW_TOKENS, EGO_MAX_PIXELS,
                         EGO_MEDIA_REVISION, EGO_SHEET_H, EGO_SHEET_W, EGO_SOURCE_SHA256, MIN_PIXELS)
from ..hashing import sha256_bytes, sha256_file, sha256_text
from ..io_utils import read_json, read_jsonl, write_once_bytes, write_once_json
from .errors import ImageIdentityError, MissingImageError, RuntimeContractError

PRED_SCHEMA = "ego_pred_v1"
DIAG_SCHEMA = "ego_diag_v1"
EGO_VISUAL_RANGE = (EGO_EXPECTED_VISUAL_TOKENS, EGO_EXPECTED_VISUAL_TOKENS)
COHORT_FIELD = {"profile5": "profiling_ids", "full400": "ids", "diag32": "diagnostic_ids"}
COHORT_SIZE = {"profile5": 5, "full400": 400, "diag32": 32}
MAX_CONSECUTIVE_FAILURES = 8


# ------------------------------------------------------------------ inputs (gold stripped)
@dataclass(frozen=True)
class EgoInput:
    id: str
    behaviors: tuple
    justifications: tuple


def inputs_from_items(items: Mapping, ids: Sequence[str]) -> dict:
    """Whitelist: id, behaviors, justifications. correct/sensible gold never leaves this function."""
    missing = [i for i in ids if i not in items]
    if missing:
        raise RuntimeContractError(f"{len(missing)} cohort IDs are not schema-valid source records")
    return {i: EgoInput(i, tuple(items[i].behaviors), tuple(items[i].justifications)) for i in ids}


def load_ids_manifest(path: str | Path, sha256: str) -> dict:
    if sha256_file(path) != sha256:
        raise RuntimeContractError("Ego ID manifest sha256 differs from the launch/Seal A value")
    doc = read_json(path)
    ids = doc.get("ids") or []
    if doc.get("annotation", {}).get("sha256") != EGO_SOURCE_SHA256 or doc.get("media_revision") != EGO_MEDIA_REVISION:
        raise RuntimeContractError("Ego ID manifest is not bound to the pinned source/media revision")
    if len(ids) != COHORT_SIZE["full400"] or len(set(ids)) != len(ids) or \
            doc.get("diagnostic_ids") != ids[:32] or doc.get("profiling_ids") != ids[:5]:
        raise RuntimeContractError("Ego ID manifest cohorts are not the frozen 400 / first-32 / first-5 lists")
    return doc


def load_inputs(source_path: str | Path, ids_doc: Mapping) -> dict:
    """Validate the pinned source, re-derive the frozen order by the locked rule, strip gold."""
    from ..data.egonormia import load_final_data_1853, locked_subset

    src = load_final_data_1853(source_path, verify_pin=True)
    if locked_subset(sorted(src["items"])) != list(ids_doc["ids"]):
        raise RuntimeContractError("frozen Ego IDs differ from the locked rule applied to the pinned source")
    return inputs_from_items(src["items"], ids_doc["ids"])


def cohort_ids(ids_doc: Mapping, cohort: str) -> list[str]:
    ids = list(ids_doc[COHORT_FIELD[cohort]])
    if len(ids) != COHORT_SIZE[cohort]:
        raise RuntimeContractError(f"cohort {cohort} has {len(ids)} IDs (no silent shrinking)")
    return ids


# ------------------------------------------------------------------ media
def load_sheet_manifest(path: str | Path, sha256: str, ids: Sequence[str]) -> dict:
    """Media-session sheet cache records; every cohort ID must be present (ok or decode_failed)."""
    if sha256_file(path) != sha256:
        raise RuntimeContractError("Ego sheet manifest sha256 differs from the launch/Seal A value")
    recs = {}
    for r in read_jsonl(path):
        if r["id"] in recs:
            raise RuntimeContractError(f"duplicate sheet record for {r['id']}")
        recs[r["id"]] = r
    missing = [i for i in ids if i not in recs]
    if missing:
        raise RuntimeContractError(f"{len(missing)} cohort IDs have no sheet record (no silent cohort shrink)")
    for i in ids:
        if recs[i].get("status") not in ("ok", "decode_failed"):
            raise RuntimeContractError(f"{i}: unknown sheet status {recs[i].get('status')!r}")
    return recs


class SheetLoader:
    def load(self, rec: Mapping):
        from PIL import Image

        p = Path(rec.get("sheet_path") or "")
        if not p.is_file():
            raise MissingImageError(f"{rec['id']}: frame sheet {p} missing (no fallback, never text-only)")
        if sha256_file(p) != rec.get("sheet_sha256"):
            raise ImageIdentityError(f"{rec['id']}: frame sheet sha256 differs from the sheet manifest")
        with Image.open(p) as im:
            im.load()
            img = im.convert("RGB")
        if img.size != (EGO_SHEET_W, EGO_SHEET_H):
            raise ImageIdentityError(f"{rec['id']}: sheet size {img.size} != ({EGO_SHEET_W}, {EGO_SHEET_H})")
        return img


def configure_ego_processor(processor) -> dict:
    """A5 sheet budget: max 524,288 pixels (the 1024x512 sheet is processed unresized)."""
    from .model import configure_image_budget

    return configure_image_budget(processor, MIN_PIXELS, EGO_MAX_PIXELS)


# ------------------------------------------------------------------ native generation
def _id_set(v) -> list[int]:
    if v is None:
        return []
    return sorted({int(x) for x in (v if isinstance(v, (list, tuple)) else [v])})


class NativeGenerator:
    """Greedy native generation with the model's own generation_config (only the plan's
    do_sample/max_new_tokens/use_cache are passed)."""

    def __init__(self, model, tokenizer, terminator_id: int, device: str, max_new_tokens: int = EGO_MAX_NEW_TOKENS):
        from .model import base_of

        self.model, self.tok, self.device, self.max_new = model, tokenizer, device, int(max_new_tokens)
        self.base = base_of(model)
        gc = self.base.generation_config
        self.eos_ids = _id_set(gc.eos_token_id)
        if not self.eos_ids:
            raise RuntimeContractError("native generation_config has no eos_token_id; decision needed (never guessed)")
        if getattr(gc, "num_beams", 1) not in (1, None):
            raise RuntimeContractError(f"native generation_config num_beams={gc.num_beams}; greedy protocol needs 1")
        self.terminator_id = int(terminator_id)
        self.generation_config = gc.to_dict()

    def __call__(self, ser) -> dict:
        import torch

        if self.model.training:
            raise RuntimeContractError("generation requires model.eval()")
        n = len(ser.input_ids)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in dict(ser.model_inputs).items()}
        if "past_key_values" in inputs or tuple(inputs["input_ids"].shape) != (1, n):
            raise RuntimeContractError("independent batch-1 call with a fresh cache required")
        self.base.model.rope_deltas = None  # no rope/cache state carried between independent calls
        t0 = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new, use_cache=True)
        self.base.model.rope_deltas = None
        if out[0, :n].tolist() != list(ser.input_ids):
            raise RuntimeContractError("generate output does not start with the prompt")
        new = [int(t) for t in out[0, n:].tolist()]
        return gen_record(new, self.tok, self.eos_ids, self.terminator_id, self.max_new,
                          {"prompt_sha256": ser.prompt_sha256, "num_prompt_tokens": n,
                           "num_visual_tokens": ser.num_visual_tokens, "seconds": time.monotonic() - t0})


def gen_record(new: Sequence[int], tokenizer, eos_ids: Sequence[int], terminator_id: int, max_new: int,
               meta: Mapping) -> dict:
    eos_pos = next((k for k, t in enumerate(new) if t in eos_ids), None)
    term_pos = next((k for k, t in enumerate(new) if t == terminator_id), None)
    ends = [p for p in (eos_pos, term_pos) if p is not None]
    end = min(ends) if ends else len(new)
    if len(new) > max_new:
        raise RuntimeContractError("generation exceeded max_new_tokens")
    return {
        "raw": tokenizer.decode(list(new[:end]), skip_special_tokens=False),
        "raw_full": tokenizer.decode(list(new), skip_special_tokens=False),
        "token_ids": list(new), "gen_len": len(new),
        "eos": eos_pos is not None, "eos_position": eos_pos, "native_eos_ids": list(eos_ids),
        "terminator_id": terminator_id, "terminator_position": term_pos,
        "tokens_after_turn_end": len(new) - end - 1 if ends else 0,
        "cap_hit": len(new) >= max_new and eos_pos is None,
        "max_new_tokens": max_new, **dict(meta),
    }


# ------------------------------------------------------------------ candidates (A5.1, diag only)
def candidate_sequences(tokenizer, ser, terminator_id: int) -> list[list[int]]:
    """Model-free checks: each of the 32 canonical answers tokenizes with the prompt as an exact
    prefix and decodes back to its text; + native terminator; distinct and prefix-free."""
    from ..ego.rotations import candidate_text, check_prefix_free
    from ..serializer import to_list

    prompt = [int(x) for x in ser.input_ids]
    seqs = []
    for m in range(32):
        text = candidate_text(m)
        full = [int(x) for x in to_list(tokenizer(ser.expanded + text, add_special_tokens=False)["input_ids"])]
        cand = full[len(prompt):]
        if full[:len(prompt)] != prompt or not cand or tokenizer.decode(cand) != text:
            raise RuntimeContractError(f"candidate {text!r} does not tokenize with the prompt as an exact prefix")
        if terminator_id in cand:
            raise RuntimeContractError("candidate text tokenizes into the terminator")
        seqs.append(cand + [int(terminator_id)])
    try:
        check_prefix_free(seqs)
    except ValueError as exc:
        raise RuntimeContractError(f"candidate sequences: {exc}") from exc
    return seqs


def candidate_scores(model, tokenizer, ser, terminator_id: int, device: str) -> dict:
    """Teacher-forced full-sequence log-probabilities of the 32 canonical answers + terminator.
    Scores are FP32 log_softmax summed in FP64; no length normalization."""
    import torch

    from .model import base_of

    if model.training:
        raise RuntimeContractError("candidate scoring requires model.eval()")
    prompt = [int(x) for x in ser.input_ids]
    seqs = candidate_sequences(tokenizer, ser, terminator_id)
    base = base_of(model)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(ser.model_inputs).items()}
    pv = inputs.pop("pixel_values", None)
    scores = []
    with torch.no_grad():
        mm = None
        if pv is not None:
            mm = {"image": base.model.get_image_features(pv, inputs["image_grid_thw"], return_dict=True)}
        for seq in seqs:
            ids = torch.tensor([prompt + seq], device=device)
            extra = {"attention_mask": torch.ones_like(ids)}
            if "mm_token_type_ids" in inputs:
                pad = torch.zeros(1, len(seq), dtype=inputs["mm_token_type_ids"].dtype, device=device)
                extra["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
            if "image_grid_thw" in inputs:
                extra["image_grid_thw"] = inputs["image_grid_thw"]
            base.model.rope_deltas = None
            logits = model(input_ids=ids, mm_encoder_outputs=mm, use_cache=False, logits_to_keep=len(seq) + 1,
                           **extra).logits[0, :len(seq)]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeContractError("nonfinite candidate logits")
            lp = torch.log_softmax(logits.float(), dim=-1)
            scores.append(float(sum(float(lp[k, t]) for k, t in enumerate(seq))))
    base.model.rope_deltas = None
    return candidate_result(scores, seqs, terminator_id)


def candidate_result(scores: Sequence[float], seqs: Sequence[Sequence[int]], terminator_id: int) -> dict:
    """Fail closed: exactly 32 finite summed log-probs (finite logits give finite log_softmax),
    finite log mass, and total canonical-completion mass <= 1 + tolerance."""
    if len(scores) != 32 or len(seqs) != 32 or not all(isinstance(s, float) and math.isfinite(s) for s in scores):
        raise RuntimeContractError("candidate scoring must produce 32 finite log-probabilities")
    mx = max(scores)
    log_mass = mx + math.log(sum(math.exp(s - mx) for s in scores))
    if not math.isfinite(log_mass) or math.exp(log_mass) > 1 + EGO_CANDIDATE_MASS_TOL:
        raise RuntimeContractError(f"candidate log mass {log_mass} invalid or total mass exceeds 1+tolerance")
    return {"logprobs": scores, "token_lengths": [len(s) for s in seqs], "terminator_id": int(terminator_id),
            "log_mass": log_mass, "mass": math.exp(log_mass)}


# ------------------------------------------------------------------ one item
def run_item(inp: EgoInput, sheet, serializer, generate: Callable, mode: str, arm: str, seed,
             candidates: Callable | None = None) -> dict:
    """mode 'primary' -> scorer row; mode 'diag' -> rotations + candidates only."""
    from ..ego.parsers import parse_action
    from ..ego.rotations import rotate_options
    from ..prompts import render_ego_best, render_ego_justification, render_ego_sensible

    beh, jus = list(inp.behaviors), list(inp.justifications)

    def visual(task, text):
        return serializer.ego(task, inp.id, text, sheet, text_only=False, visual_range=EGO_VISUAL_RANGE)

    if mode == "diag":
        s0 = visual("ego_sensible", render_ego_sensible(beh))
        rot = {str(r): generate(visual("ego_sensible", render_ego_sensible(rotate_options(beh, r)))) for r in (1, 2, 3, 4)}
        return {"id": inp.id, "arm": arm, "seed": seed, "schema": DIAG_SCHEMA, "rotations": rot,
                "candidates": candidates(s0) if candidates else None,
                "original_prompt_sha256": s0.prompt_sha256}
    best = generate(visual("ego_best", render_ego_best(beh)))
    ap = parse_action(best["raw"])
    just = None
    if ap.valid:  # model-selected behavior only; invalid action skips only this dependent call
        just = generate(visual("ego_justification", render_ego_justification(beh[ap.index], jus)))
    sensible = generate(visual("ego_sensible", render_ego_sensible(beh)))
    text_only = generate(serializer.ego("ego_sensible_text_only", inp.id, render_ego_sensible(beh, text_only=True),
                                        None, text_only=True))
    return {"id": inp.id, "arm": arm, "seed": seed, "schema": PRED_SCHEMA, "best": best, "justification": just,
            "justification_selected_index": ap.index if ap.valid else None, "best_parse_reason": ap.reason,
            "sensible": sensible, "sensible_text_only": text_only}


# ------------------------------------------------------------------ resumable run
class EgoRun:
    """Per-ID write-once items/failures; finalize only when every cohort ID is accounted for."""

    def __init__(self, out_dir: str | Path, identity: Mapping, ids: Sequence[str], decode_failed: set,
                 process: Callable[[str], dict], row_errors: tuple, budget=None, first_item_reserve_s: float = 0.0):
        self.out = Path(out_dir)
        self.identity, self.ids, self.decode_failed = dict(identity), list(ids), set(decode_failed)
        if not set(self.decode_failed) <= set(self.ids) or len(set(self.ids)) != len(self.ids):
            raise RuntimeContractError("cohort IDs must be unique and decode failures a subset")
        self.process, self.row_errors, self.budget = process, row_errors, budget
        self.reserve = float(first_item_reserve_s)

    def _path(self, kind: str, i: str) -> Path:
        return self.out / kind / f"{sha256_text(i)[:32]}.json"

    def open(self) -> None:
        if (self.out / "identity.json").exists():
            if read_json(self.out / "identity.json", verify=True) != self.identity:
                raise RuntimeContractError("existing Ego run directory belongs to a different run identity")
        else:
            write_once_json(self.out / "identity.json", self.identity)

    def state(self, i: str) -> str | None:
        """'item' | 'failure' | None. Both present is a corrupted run and is refused, never resolved."""
        item, fail = self._path("items", i).exists(), self._path("failures", i).exists()
        if item and fail:
            raise RuntimeContractError(f"{i}: both an item and a failure artifact exist; refusing to choose")
        return "item" if item else "failure" if fail else None

    def done(self, i: str) -> bool:
        return self.state(i) is not None

    def run(self) -> dict:
        consecutive = 0
        for i in self.ids:
            if i in self.decode_failed or self.done(i):
                continue
            if self.budget is not None and not self.budget.allows(self.reserve):
                return {"status": "paused", "reason": "time cap", "remaining": self.remaining()}
            t0 = time.monotonic()
            try:
                row = self.process(i)
                if row.get("id") != i:
                    raise RuntimeContractError(f"item row id {row.get('id')!r} != {i!r}")
                write_once_json(self._path("items", i), row)
                consecutive = 0
            except self.row_errors as exc:
                write_once_json(self._path("failures", i), {"id": i, "error_type": type(exc).__name__,
                                                             "error": str(exc)})
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeContractError(f"{consecutive} consecutive Ego item failures; systematic, stopping")
            self.reserve = max(self.reserve, time.monotonic() - t0)
        return {"status": "items_complete", "remaining": 0}

    def remaining(self) -> int:
        return sum(1 for i in self.ids if i not in self.decode_failed and not self.done(i))

    def finalize(self, provenance: Mapping) -> dict:
        """Deterministic, restart-safe: the exact bytes of predictions.jsonl, failures.jsonl and
        manifest.json are derived from the verified per-ID artifacts and identity only. Each file is
        written once, or, if it already exists (e.g. a crash mid-finalize), verified byte-for-byte
        against the desired bytes and its sidecar; any difference stops and nothing is replaced.
        Per-attempt provenance goes to a new write-once finalize_receipt_NNN.json."""
        if self.remaining():
            raise RuntimeContractError(f"{self.remaining()} cohort IDs unaccounted for; cannot finalize")
        rows, fails = [], []
        for i in self.ids:
            if i in self.decode_failed:
                continue
            kind = self.state(i)
            r = read_json(self._path("items" if kind == "item" else "failures", i), verify=True)
            if r.get("id") != i:
                raise RuntimeContractError(f"{kind} file for {i} holds {r.get('id')!r}")
            (rows if kind == "item" else fails).append(r)
        pred = _jsonl_bytes(rows)
        fail = _jsonl_bytes(fails)
        manifest = {"schema": "gn_ego_run_v1", "identity": self.identity, "cohort_ids": self.ids,
                    "decode_failed": sorted(self.decode_failed), "predicted": len(rows), "failed": len(fails),
                    "complete": not fails, "predictions_sha256": sha256_bytes(pred),
                    "failures_sha256": sha256_bytes(fail),
                    "note": "decode failures are the common pre-score exclusion; failures make the run incomplete "
                            "and the scorer refuses missing rows (never replaced)"}
        _write_or_verify(self.out / "predictions.jsonl", pred)
        _write_or_verify(self.out / "failures.jsonl", fail)
        _write_or_verify(self.out / "manifest.json", _json_bytes(manifest))
        n = len(list(self.out.glob("finalize_receipt_*.json")))
        write_once_json(self.out / f"finalize_receipt_{n:03d}.json",
                        {"manifest_sha256": sha256_file(self.out / "manifest.json"), "provenance": dict(provenance)})
        return manifest


# Byte formats identical to io_utils.write_once_jsonl / write_once_json.
def _jsonl_bytes(rows: list[dict]) -> bytes:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_or_verify(path: Path, data: bytes) -> str:
    """Write once; or verify an existing artifact equals `data` exactly (completing only a missing
    sidecar for identical bytes, as left by a crash inside write_once_bytes). Never replaces."""
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


def load_run_rows(manifest_path: str | Path, expected_manifest_sha256: str) -> tuple[dict, list[dict]]:
    """Strict downstream join helper: manifest hash (launch/Seal value) + sidecar, then the
    predictions file hash/sidecar must equal the manifest, and the rows must be exactly the
    cohort minus decode failures minus failures, in cohort order, one per ID."""
    manifest_path = Path(manifest_path)
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeContractError("Ego run manifest sha256 differs from the expected value")
    man = read_json(manifest_path, verify=True)
    d = manifest_path.parent
    for name, key in (("predictions.jsonl", "predictions_sha256"), ("failures.jsonl", "failures_sha256")):
        if sha256_file(d / name) != man[key]:
            raise RuntimeContractError(f"{name} sha256 differs from its manifest")
    rows = read_jsonl(d / "predictions.jsonl", verify=True)
    fails = read_jsonl(d / "failures.jsonl", verify=True)
    cohort, excluded = list(man["cohort_ids"]), list(man["decode_failed"])
    fail_ids = [f.get("id") for f in fails]
    problems = []
    if len(set(cohort)) != len(cohort) or len(set(excluded)) != len(excluded) or not set(excluded) <= set(cohort):
        problems.append("cohort/decode-failure lists invalid")
    if len(set(fail_ids)) != len(fail_ids) or not set(fail_ids) <= set(cohort) - set(excluded):
        problems.append("failure IDs duplicated or outside the evaluated cohort")
    want = [i for i in cohort if i not in set(excluded) and i not in set(fail_ids)]
    if [r.get("id") for r in rows] != want:
        problems.append("prediction rows are not exactly the evaluated cohort minus failures, in cohort order")
    if len(rows) != man["predicted"] or len(fails) != man["failed"] or man["complete"] is not (not fails):
        problems.append("manifest counts/complete flag inconsistent with the artifacts")
    if problems:
        raise RuntimeContractError("Ego run join refused: " + "; ".join(problems))
    return man, rows


# Identity fields a diag32 run must share with its paired full400 primary run.
PAIRED_IDENTITY_KEYS = ("runtime_version", "phase", "arm", "seed", "source_sha256", "ids_manifest_sha256",
                        "sheet_manifest_sha256", "pins_sha256", "seal_a_sha256", "adapter_dir",
                        "adapter_export_report_sha256", "max_new_tokens", "max_prompt_tokens", "template_sha256")


def check_diag_pairing(primary_manifest: Mapping, diag_identity: Mapping, ids_doc: Mapping,
                       sheets: Mapping) -> None:
    """A diag32 run may only pair with the exact complete full400 run of the same configuration."""
    pi = primary_manifest.get("identity", {})
    diff = [k for k in PAIRED_IDENTITY_KEYS if pi.get(k) != diag_identity.get(k)]
    problems = []
    if diff:
        problems.append(f"identity differs in {diff}")
    if pi.get("cohort") != "full400" or primary_manifest.get("complete") is not True:
        problems.append("primary must be a complete full400 run")
    if primary_manifest.get("cohort_ids") != list(ids_doc["ids"]):
        problems.append("primary cohort IDs/order differ from the frozen 400")
    full_failed = sorted(i for i in ids_doc["ids"] if sheets[i]["status"] == "decode_failed")
    if primary_manifest.get("decode_failed") != full_failed:
        problems.append("primary decode-failure exclusions differ from the sheet manifest policy")
    if problems:
        raise RuntimeContractError("diag32 pairing refused: " + "; ".join(problems))


# ------------------------------------------------------------------ GPU entry point
def ego(launch_path: str) -> dict:
    """`python -m grounded_norm_sft.runtime ego --launch L` (ego_profile_preseal or ego_eval)."""
    from ..data.egonormia import EgoSchemaError
    from ..prompts import verify_ego_template_hashes
    from ..serializer import QuarantineError, SerializationError, Serializer
    from . import RUNTIME_VERSION
    from .export import assert_native_source
    from .gates import deadline_seconds, load_launch
    from .runner import _load_verified_adapter, _stack, setup
    from .train import Budget

    launch, _ = load_launch(launch_path)
    if launch.get("ego", {}).get("diagnostic_protocol") is not None:
        raise RuntimeContractError("generation protocol requires the distinct ego-generation-diagnostic command")
    phase = launch.get("phase")
    if phase not in ("ego_profile_preseal", "ego_eval"):
        raise RuntimeContractError(f"ego runs ego_profile_preseal/ego_eval launches, not {phase!r}")
    ctx = setup(launch_path, phase)
    budget = Budget(deadline_seconds(ctx.launch))
    e, arm, seed = ctx.launch["ego"], ctx.launch["arm"], ctx.launch.get("seed")
    templates = verify_ego_template_hashes()
    ids_doc = load_ids_manifest(e["ids_manifest_path"], e["ids_manifest_sha256"])
    ids = cohort_ids(ids_doc, e["cohort"])
    inputs = load_inputs(e["source_path"], ids_doc)
    mode = "diag" if e["cohort"] == "diag32" else "primary"
    # full frozen list for diag so the primary's decode-failure policy can be checked too
    sheets = load_sheet_manifest(e["sheet_manifest_path"], e["sheet_manifest_sha256"],
                                 ids_doc["ids"] if mode == "diag" else ids)
    decode_failed = {i for i in ids if sheets[i]["status"] == "decode_failed"}
    adapter_report = (sha256_file(Path(e["adapter_dir"]) / "export_report.json")
                      if arm == "B1" else None)
    identity = {"runtime_version": RUNTIME_VERSION, "phase": phase, "arm": arm, "seed": seed, "cohort": e["cohort"],
                "source_sha256": e["source_sha256"], "ids_manifest_sha256": e["ids_manifest_sha256"],
                "sheet_manifest_sha256": e["sheet_manifest_sha256"], "pins_sha256": ctx.pins_sha256,
                "seal_a_sha256": (ctx.launch.get("seal_a") or {}).get("sha256"),
                "adapter_dir": e.get("adapter_dir"), "adapter_export_report_sha256": adapter_report,
                "primary_manifest_sha256": e.get("primary_manifest_sha256"),
                "max_new_tokens": EGO_MAX_NEW_TOKENS, "max_prompt_tokens": e["max_prompt_tokens"],
                "template_sha256": templates}
    if mode == "diag":
        prim, _ = load_run_rows(e["primary_manifest_path"], e["primary_manifest_sha256"])  # artifact hashes
        check_diag_pairing(prim, identity, ids_doc, sheets)
    out = ctx.out_root / f"ego_{phase}_{arm}_{e['cohort']}_{sha256_text(repr(sorted(identity.items())))[:16]}"

    st = _stack(ctx)
    model = st["model"]
    extra = {}
    if arm == "B1":
        model, extra = _load_verified_adapter(ctx, model, Path(e["adapter_dir"]))
    else:
        extra = {"native_source": assert_native_source(model)}
    model.eval()
    processor = st["processor"]
    budget_report = configure_ego_processor(processor)
    serializer = Serializer(processor, max_tokens=int(e["max_prompt_tokens"]))
    term = st["processor_report"]["tokens"]["assistant_turn_terminator"]["id"]
    gen = NativeGenerator(model, processor.tokenizer, term, "cuda")
    cand = (lambda s: candidate_scores(model, processor.tokenizer, s, term, "cuda")) if mode == "diag" else None
    loader = SheetLoader()

    def process(i):
        return run_item(inputs[i], loader.load(sheets[i]), serializer, gen, mode, arm, seed, cand)

    run = EgoRun(out, identity, ids, decode_failed, process,
                 (MissingImageError, ImageIdentityError, QuarantineError, SerializationError, EgoSchemaError),
                 budget=budget, first_item_reserve_s=60.0 * float(e["first_item_reserve_minutes"]))
    run.open()
    res = run.run()
    if res["status"] == "paused":
        return {**res, "out": str(out)}
    manifest = run.finalize({**ctx.provenance(), "native_generation_config": gen.generation_config,
                             "native_eos_ids": gen.eos_ids, "assistant_turn_terminator": term,
                             "image_budget": budget_report, "adapter": extra, "device": st["device"],
                             "profile_only_no_accuracy": phase == "ego_profile_preseal"})
    return {"status": "complete" if manifest["complete"] else "incomplete", "out": str(out),
            "predicted": manifest["predicted"], "failed": manifest["failed"],
            "decode_failed": len(manifest["decode_failed"])}
