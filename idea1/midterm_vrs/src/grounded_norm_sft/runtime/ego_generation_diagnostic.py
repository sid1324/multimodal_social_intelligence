"""Prospective first32 generation comparison; never a primary/candidate scorer artifact."""
from __future__ import annotations

import json
from pathlib import Path

from ..hashing import sha256_file, sha256_text
from ..io_utils import read_json, write_once_json
from ..prompts import render_ego_sensible
from . import ego as E
from .errors import RuntimeContractError

PROTOCOL = "ego_generation_first32_v1"
SCHEMA = "ego_generation_diagnostic_v1"
SUFFIX = ("\nFor this response, override the earlier requests to show reasoning and the reasoning example. "
          "Output only one Python list of the sensible option indices from 1 to 5, or [] if no option is sensible. "
          "Do not include explanations or any other text.")
CONDITIONS = ("original256_reused", "cap512_only", "concise256_only")
TASKS = ("sensible", "sensible_text_only")
AUTHORITY_SUMMARY = (
    "Trusted user requests bounded generation-protocol comparison on the same frozen first32 Ego IDs, "
    "reusing originalprompt256 controls and adding increased-cap-only and concise-format-only, with all "
    "other decoding matched across B0/B1/later methods; preserve originals/report all conditions "
    "parse/cap/task/cost; never select protocol by accuracy.")


def validate_protocol(doc):
    expected = protocol_document()
    if doc != expected:
        raise RuntimeContractError("generation diagnostic protocol differs from frozen v1")
    return doc


def protocol_document():
    # Deliberately fixed, with no user-supplied prompt/cap/search space.
    return {"schema": PROTOCOL, "suffix": SUFFIX, "suffix_sha256": sha256_text(SUFFIX),
            "conditions": list(CONDITIONS), "tasks": list(TASKS), "caps": [256, 512, 256],
            "max_ids": 32, "cohort_rule": "diagnostic_ids == ids[:32]", "selection_by_accuracy": False,
            "authority_phase": "ego_eval", "authority_summary": AUTHORITY_SUMMARY,
            "authority_summary_is_verbatim": False,
            "authority_receipt_recorded_at_utc": "2026-10-24T02:04:41Z",
            "original_utterance_time": "unavailable; receipt time is not utterance time",
            "authority_source_chat": "01a16ae0-36f1-718d-8a6c-64158e6db02f",
            "decoding": "native greedy; do_sample=False; use_cache=True; native EOS and terminator unchanged",
            "scope": "sensible set visual and text only; best and justification unchanged; no rotations or candidate scores",
            "order": "condition-major: reused control, cap-only, concise-only; frozen ID order within condition",
            "cost_bound": "one H100; at most 6 hours/job; existing aggregate 100 GPU-hour authority"}


def serialize_tasks(inp, sheet, serializer, concise=False):
    result = {}
    for task in TASKS:
        text_only = task == "sensible_text_only"
        text = render_ego_sensible(list(inp.behaviors), text_only=text_only)
        result[task] = serializer.ego("ego_" + task, inp.id, text + (SUFFIX if concise else ""),
                                      None if text_only else sheet, text_only=text_only,
                                      **({} if text_only else {"visual_range": E.EGO_VISUAL_RANGE}))
    return result


def check_control(inp, serialized, primary):
    if primary.get("schema") != E.PRED_SCHEMA or primary.get("id") != inp.id:
        raise RuntimeContractError("control must be an exact primary row")
    for task in TASKS:
        r, s = primary[task], serialized[task]
        if r.get("max_new_tokens") != 256 or r.get("prompt_sha256") != s.prompt_sha256 or \
                r.get("num_prompt_tokens") != len(s.input_ids) or r.get("num_visual_tokens") != s.num_visual_tokens:
            raise RuntimeContractError(f"{inp.id}/{task}: reused control prompt/cap/visual budget differs")


def condition_row(inp, sheet, serializer, generate, condition, primary, arm, seed):
    if condition not in CONDITIONS:
        raise RuntimeContractError("unknown frozen diagnostic condition")
    originals = serialize_tasks(inp, sheet, serializer)
    check_control(inp, originals, primary)
    if primary.get("arm") != arm or primary.get("seed") != seed:
        raise RuntimeContractError("reused primary arm/seed differs")
    prompts = serialize_tasks(inp, sheet, serializer, concise=True) if condition == CONDITIONS[2] else originals
    results = {t: primary[t] if condition == CONDITIONS[0] else generate(prompts[t]) for t in TASKS}
    for t, r in results.items():
        control = primary[t]
        cap = 512 if condition == CONDITIONS[1] else 256
        if r.get("max_new_tokens") != cap or r.get("prompt_sha256") != prompts[t].prompt_sha256:
            raise RuntimeContractError("condition cap/prompt differs from frozen protocol")
        for k in ("native_eos_ids", "terminator_id"):
            if r.get(k) != control.get(k):
                raise RuntimeContractError(f"native {k} differs from control")
    return {"id": inp.id, "schema": SCHEMA, "arm": arm, "seed": seed, "condition": condition,
            "reused": condition == CONDITIONS[0], "results": results,
            "original_prompt_sha256": {t: originals[t].prompt_sha256 for t in TASKS}}


def ego_generation_diagnostic(launch_path):
    from ..data.egonormia import EgoSchemaError
    from ..prompts import verify_ego_template_hashes
    from ..serializer import Serializer, QuarantineError, SerializationError
    from . import RUNTIME_VERSION
    from .export import assert_native_source
    from .gates import deadline_seconds, load_launch
    from .runner import setup, _stack, _load_verified_adapter
    from .train import Budget

    launch, _ = load_launch(launch_path)
    e = launch.get("ego", {})
    if launch.get("phase") != "ego_eval" or e.get("diagnostic_protocol") != PROTOCOL or e.get("cohort") != "diag32":
        raise RuntimeContractError("distinct diagnostic command requires explicit ego_eval/diag32 protocol")
    ctx = setup(launch_path, "ego_eval")
    protocol_path = e["protocol_manifest_path"]
    if sha256_file(protocol_path) != e["protocol_manifest_sha256"]:
        raise RuntimeContractError("protocol manifest digest differs")
    protocol = validate_protocol(read_json(protocol_path, verify=True))
    ids_doc = E.load_ids_manifest(e["ids_manifest_path"], e["ids_manifest_sha256"])
    ids = E.cohort_ids(ids_doc, "diag32")
    inputs = E.load_inputs(e["source_path"], ids_doc)
    sheets = E.load_sheet_manifest(e["sheet_manifest_path"], e["sheet_manifest_sha256"], ids_doc["ids"])
    primary, rows = E.load_run_rows(e["primary_manifest_path"], e["primary_manifest_sha256"])
    arm, seed = launch["arm"], launch.get("seed")
    identity = {"runtime_version": RUNTIME_VERSION, "phase": "ego_eval", "arm": arm, "seed": seed, "cohort": "diag32",
                "source_sha256": e["source_sha256"], "ids_manifest_sha256": e["ids_manifest_sha256"],
                "sheet_manifest_sha256": e["sheet_manifest_sha256"], "pins_sha256": ctx.pins_sha256,
                "seal_a_sha256": launch["seal_a"]["sha256"], "adapter_dir": e.get("adapter_dir"),
                "adapter_export_report_sha256": sha256_file(Path(e["adapter_dir"]) / "export_report.json") if arm == "B1" else None,
                "primary_manifest_sha256": e["primary_manifest_sha256"], "max_new_tokens": 256,
                "max_prompt_tokens": e["max_prompt_tokens"], "template_sha256": verify_ego_template_hashes(),
                "diagnostic_protocol": PROTOCOL, "protocol_manifest_sha256": e["protocol_manifest_sha256"]}
    E.check_diag_pairing(primary, identity, ids_doc, sheets)
    controls = {r["id"]: r for r in rows}
    budget = Budget(deadline_seconds(launch))
    st = _stack(ctx)
    model = st["model"]
    if arm == "B1":
        model, extra = _load_verified_adapter(ctx, model, Path(e["adapter_dir"]))
    else:
        extra = {"native_source": assert_native_source(model)}
    model.eval()
    processor = st["processor"]
    image_budget = E.configure_ego_processor(processor)
    serializer = Serializer(processor, max_tokens=int(e["max_prompt_tokens"]))
    term = st["processor_report"]["tokens"]["assistant_turn_terminator"]["id"]
    gens = {256: E.NativeGenerator(model, processor.tokenizer, term, "cuda", 256),
            512: E.NativeGenerator(model, processor.tokenizer, term, "cuda", 512)}
    out = ctx.out_root / (PROTOCOL + "_" + sha256_text(repr(sorted(identity.items())))[:16])
    prompt_bindings = {}
    for i in ids:
        if sheets[i]["status"] == "decode_failed":
            continue
        sheet = E.SheetLoader().load(sheets[i])
        try:
            original = serialize_tasks(inputs[i], sheet, serializer)
            check_control(inputs[i], original, controls[i])
            concise = serialize_tasks(inputs[i], sheet, serializer, concise=True)
        finally:
            if hasattr(sheet, "close"):
                sheet.close()
        prompt_bindings[i] = {"sheet_sha256": sheets[i]["sheet_sha256"],
                              "conditions": {c: {t: (concise if c == CONDITIONS[2] else original)[t].prompt_sha256
                                                  for t in TASKS} for c in CONDITIONS}}
    receipts = sorted(Path(e["primary_manifest_path"]).parent.glob("finalize_receipt_*.json"))
    if not receipts:
        raise RuntimeContractError("primary native-generation provenance receipt is required")
    receipt = read_json(receipts[0], verify=True)
    if receipt.get("manifest_sha256") != e["primary_manifest_sha256"] or \
            receipt.get("provenance", {}).get("native_generation_config") != gens[256].generation_config:
        raise RuntimeContractError("native generation settings differ from primary receipt")
    frozen = {"protocol": protocol, "identity": identity, "cohort_ids": ids, "prompt_bindings": prompt_bindings,
              "primary_receipt_sha256": sha256_file(receipts[0]),
              "control_rows_sha256": {i: sha256_text(json.dumps(controls[i], sort_keys=True))
                                       for i in ids if i in controls}}
    freeze = out / "frozen_comparison.json"
    if freeze.exists():
        if read_json(freeze, verify=True) != frozen:
            raise RuntimeContractError("frozen comparison differs on resume")
    else:
        write_once_json(freeze, frozen)  # before the first diagnostic generation
    failed = {i for i in ids if sheets[i]["status"] == "decode_failed"}
    for condition in CONDITIONS:
        generate = gens[512 if condition == CONDITIONS[1] else 256]
        def process(i):
            sheet = E.SheetLoader().load(sheets[i])
            try:
                return condition_row(inputs[i], sheet, serializer, generate, condition, controls[i], arm, seed)
            finally:
                if hasattr(sheet, "close"):
                    sheet.close()
        run = E.EgoRun(out / condition, {**identity, "condition": condition}, ids, failed, process,
                       (E.MissingImageError, E.ImageIdentityError, QuarantineError, SerializationError, EgoSchemaError),
                       budget, 60 * float(e["first_item_reserve_minutes"]))
        run.open()
        result = run.run()
        if result["status"] == "paused":
            return {**result, "out": str(out), "condition": condition}
        manifest = run.finalize({**ctx.provenance(), "diagnostic_only": True,
                                 "native_generation_config": generate.generation_config,
                                 "image_budget": image_budget, "adapter": extra})
        if not manifest["complete"]:
            return {"status": "incomplete", "out": str(out), "condition": condition}
    return {"status": "complete", "out": str(out), "diagnostic_only": True}
