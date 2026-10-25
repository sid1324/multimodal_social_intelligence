"""GPU phase orchestration. Order is always: offline env -> pins + launch gates (metadata only,
GateError before anything heavy) -> single-use ledger reservation marked started -> device
check -> verified local load -> work -> write-once artifacts with provenance.
Nothing in this module has been executed against real weights.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..constants import NORMBANK_LETTERS, NORMLENS_LETTERS, POTENCY_FLOOR_INPUTS, POTENCY_FLOOR_REPEATS, POTENCY_SORT_PREFIX
from ..hashing import compact_json_bytes, sha256_bytes, sha256_file, sort_key_sha256
from ..io_utils import read_json, verify_sidecar, write_once_json
from . import RUNTIME_VERSION
from .errors import CheckpointError, ExportError, RuntimeContractError
from .gates import check_launch, deadline_seconds, load_launch
from .model import prepare_process_env
from .pins import load_pins

EXPORT_FIXTURE_PREFIX = POTENCY_SORT_PREFIX  # same label-free order as the plan's first-4 floor fixtures


@dataclass
class Ctx:
    phase: str
    launch: dict
    launch_sha256: str
    pins: dict
    pins_sha256: str
    gate: dict
    reservation: dict

    @property
    def out_root(self) -> Path:
        return Path(self.launch["paths"]["output_root"])

    @property
    def model_dir(self) -> str:
        return self.launch["paths"]["model_dir"]

    def provenance(self) -> dict:
        seal = self.launch.get("seal_a") or {}
        return {"runtime_version": RUNTIME_VERSION, "phase": self.phase, "launch_sha256": self.launch_sha256,
                "pins_sha256": self.pins_sha256, "seal_a_sha256": seal.get("sha256"), "gate": self.gate,
                "reservation": self.reservation, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def setup(launch_path: str, phase: str) -> Ctx:
    prepare_process_env()
    pins, pins_sha = load_pins()
    launch, launch_sha = load_launch(launch_path)
    gate = check_launch(launch, phase, pins, pins_sha, env=os.environ)
    from .gates import PHASES
    from .ledger import load_approval, mark_started

    auth = launch["authorization"]
    approval = load_approval(auth["approval_path"], auth["approval_sha256"], PHASES, pins["psc"].get("approved_su_budget"))
    started = mark_started(approval, auth["approval_sha256"], auth["reservation_id"], phase,
                           launch["job"]["time_limit_minutes"], os.environ["SLURM_JOB_ID"])
    return Ctx(phase, launch, launch_sha, pins, pins_sha, gate, started)


def _stack(ctx: Ctx) -> dict:
    """Device + determinism + processor (from the pinned snapshot) + verified base model."""
    from ..serializer import Serializer
    from .model import check_device, excluded_answer_ids, load_model, load_processor, set_determinism

    device = check_device()
    det = set_determinism()
    processor, proc_rep = load_processor(ctx.model_dir, ctx.pins, require_pin=True)
    model, model_rep = load_model(ctx.model_dir, ctx.pins, device="cuda")
    return {"device": device, "determinism": det, "processor": processor, "processor_report": proc_rep,
            "excluded_ids": excluded_answer_ids(proc_rep["tokens"]), "serializer": Serializer(processor),
            "model": model, "model_report": model_rep}


def _data(ctx: Ctx, normlens: bool = True, normbank: bool = True) -> dict:
    from .data_io import ImageLoader, load_normbank, load_normlens

    d = ctx.launch["data"]
    out = {}
    if normlens:
        out["normlens"] = load_normlens(d["normlens_manifest"], d["normlens_manifest_sha256"])
        out["images"] = ImageLoader(d["normlens_image_dir"])
    if normbank:
        out["normbank"] = load_normbank(d["normbank_manifest"], d["normbank_manifest_sha256"])
    return out


def _json_sha(obj) -> str:
    return sha256_bytes(compact_json_bytes(obj))


# ------------------------------------------------------------------ training
def _run_dev_eval(model, out_dir: Path, identity: Mapping, epoch: int, ck: Mapping, records, logits_fn, labels,
                  *, budget, reserve_s: float, provenance: Mapping) -> dict:
    """Resume only missing dev IDs for this immutable epoch checkpoint; always restore SFT mode."""
    from ..io_utils import read_jsonl
    from .evaluate import ResumableNativeEval, dev_cond_nll
    from .model import set_train_mode

    ids = [r["example_id"] for r in records]
    dev_identity = {"phase": "b1_train_dev", "training_identity": dict(identity), "epoch": epoch,
                    "checkpoint": dict(ck), "task": "normlens", "split": "dev", "condition": "image",
                    "answer_letters": list(NORMLENS_LETTERS), "expected_ids_sha256": _json_sha(ids)}
    model.eval()
    try:
        evaluation = ResumableNativeEval(out_dir, dev_identity, records, logits_fn, NORMLENS_LETTERS,
                                         seed=identity["seed"], condition="image", task="normlens", arm="B1",
                                         budget=budget, first_forward_reserve_s=reserve_s)
        evaluation.open()
        result = evaluation.run()
        if result["status"] == "paused":
            return result
        manifest = evaluation.finalize({**provenance, "checkpoint": dict(ck)})
        if not manifest["complete"]:
            raise RuntimeContractError(f"dev evaluation for epoch {epoch} incomplete ({manifest['failed']} failures)")
        rows = read_jsonl(out_dir / "predictions.jsonl", verify=True)
        return {"cond_nll": dev_cond_nll(rows, labels, ids), "predictions_sha256": manifest["predictions_sha256"]}
    finally:
        set_train_mode(model)


def train(launch_path: str, resume: bool) -> dict:
    ctx = setup(launch_path, "b1_train")
    from ..schedules import build_training_schedule
    from .data_io import normbank_input, normbank_target, normlens_input, normlens_target, scorer_labels, split_rows
    from .lora import apply_lora
    from .model import NativeForward, TaskForward, frozen_state_sha256, set_train_mode
    from .train import LR_SCHEDULE_VERSION, B1Trainer, Budget, TrainingRun

    seed, epochs = ctx.launch["seed"], ctx.launch["horizon"]["epochs"]
    budget = Budget(deadline_seconds(ctx.launch))  # started before data/model load: load time counts
    data = _data(ctx)
    nl_train, nl_dev = split_rows(data["normlens"], "train"), split_rows(data["normlens"], "dev")
    nb_train = split_rows(data["normbank"], "train")
    schedule = build_training_schedule([r["example_id"] for r in nl_train], [r["example_id"] for r in nb_train],
                                       seed, epochs, "B1")
    identity = {"arm": "B1", "seed": seed, "epochs": epochs, "schedule_sha256": _json_sha(schedule),
                "pins_sha256": ctx.pins_sha256, "seal_a_sha256": ctx.launch["seal_a"]["sha256"],
                "normlens_manifest_sha256": ctx.launch["data"]["normlens_manifest_sha256"],
                "normbank_manifest_sha256": ctx.launch["data"]["normbank_manifest_sha256"],
                "runtime_version": RUNTIME_VERSION, "lr_schedule_version": LR_SCHEDULE_VERSION}
    run_dir = ctx.out_root / f"b1_seed{seed}_e{epochs}_{_json_sha(identity)[:16]}"

    st = _stack(ctx)
    model, lora_inv = apply_lora(st["model"], seed)
    identity["lora_init_sha256"] = lora_inv["init_sha256"]
    frozen = frozen_state_sha256(model)
    native = NativeForward(model, "cuda")
    fwd = TaskForward(st["serializer"], native, data["images"], grad=True, excluded_ids=st["excluded_ids"])
    fwd_eval = TaskForward(st["serializer"], native, data["images"], grad=False, excluded_ids=st["excluded_ids"])
    trainer = B1Trainer([(n, p) for n, p in model.named_parameters() if p.requires_grad], fwd, schedule,
                        {r["example_id"]: normlens_input(r) for r in nl_train},
                        {r["example_id"]: normlens_target(r) for r in nl_train},
                        {r["example_id"]: normbank_input(r) for r in nb_train},
                        {r["example_id"]: normbank_target(r) for r in nb_train})
    dev_records = [normlens_input(r) for r in nl_dev]
    labels = scorer_labels(nl_dev)
    # The launch has only a whole-dev reserve. Reuse that conservative upper bound for
    # each per-ID check too (not a per-item timing estimate); do not infer a smaller
    # first-forward reserve by dividing it across examples. This may pause early.
    dev_reserve_s = 60.0 * float(ctx.launch["job"]["dev_reserve_minutes"])

    def dev_fn(epoch, ck):
        return _run_dev_eval(model, run_dir / f"dev_epoch{epoch}", identity, epoch, ck, dev_records,
                             fwd_eval.with_meta("normlens"), labels, budget=budget, reserve_s=dev_reserve_s,
                             provenance=ctx.provenance())

    run = TrainingRun(run_dir, identity, trainer, dev_fn, budget=budget, dev_reserve_s=dev_reserve_s)
    run.open(resume)
    receipt = {**ctx.provenance(), "identity": identity, "lora": lora_inv, "frozen_state": frozen,
               "device": st["device"], "determinism": st["determinism"], "processor": st["processor_report"],
               "model": st["model_report"], "schedule_counts": {"updates": schedule["total_updates"],
                                                                "windows": schedule["window_sizes"]}}
    if not resume:
        write_once_json(run_dir / "schedule.json", schedule)
        write_once_json(run_dir / "frozen_state.json", frozen)
        write_once_json(run_dir / "launch_receipt.json", receipt)
    else:
        if read_json(run_dir / "frozen_state.json", verify=True) != frozen:
            raise CheckpointError("frozen base weights differ from the original run")
        n = len(list(run_dir.glob("resume_receipt_*.json")))
        write_once_json(run_dir / f"resume_receipt_{n:03d}.json", receipt)
    set_train_mode(model)
    result = run.run()
    if result["status"] != "complete":
        return {**result, "run_dir": str(run_dir)}
    if frozen_state_sha256(model) != frozen:
        raise RuntimeContractError("frozen base weights changed during training")
    selection = run.select()
    return {"status": "complete", "run_dir": str(run_dir), "selection": selection, "frozen_weights_unchanged": True}


# ------------------------------------------------------------------ profiling
class _SyntheticImages:
    def load(self, rec):
        from PIL import Image

        h = hashlib.sha256(rec["example_id"].encode()).digest()
        return Image.new("RGB", (512, 384), (h[0], h[1], h[2]))


def profile(launch_path: str) -> dict:
    ctx = setup(launch_path, "gpu_profile_preseal")
    import torch

    from ..schedules import build_training_schedule
    from .data_io import normbank_input, normbank_target, normlens_input, normlens_target, split_rows
    from .lora import apply_lora
    from .model import NativeForward, TaskForward, set_train_mode
    from .train import B1Trainer, Budget

    p, seed = ctx.launch["profile"], ctx.launch["seed"]
    budget = Budget(deadline_seconds(ctx.launch))
    n = 16 * int(p["max_updates"])
    if p["scope"] == "synthetic":
        nl = {f"synthetic-{i:04d}": {"example_id": f"synthetic-{i:04d}", "image": None, "rgb_sha256": None,
                                     "action": "Hold the door open for the person behind you."} for i in range(n)}
        nb = {f"synthetic-nb-{i:04d}": {"example_id": f"synthetic-nb-{i:04d}", "setting": "an office",
                                        "behavior": "eating lunch at the desk", "constraints": "a meeting is ongoing"}
              for i in range(n)}
        nl_q = {k: [0.2, 0.5, 0.3] for k in nl}
        nb_q = {k: [0.1, 0.6, 0.3] for k in nb}
        images = _SyntheticImages()
    else:  # train split only; dev/test are never read in pre-seal profiling
        data = _data(ctx)
        tr = split_rows(data["normlens"], "train")[:n]
        nbt = split_rows(data["normbank"], "train")[:n]
        nl, nl_q = {r["example_id"]: normlens_input(r) for r in tr}, {r["example_id"]: normlens_target(r) for r in tr}
        nb, nb_q = {r["example_id"]: normbank_input(r) for r in nbt}, {r["example_id"]: normbank_target(r) for r in nbt}
        images = data["images"]
    schedule = build_training_schedule(sorted(nl), sorted(nb), seed, 1, "B1")
    st = _stack(ctx)
    model, lora_inv = apply_lora(st["model"], seed)
    set_train_mode(model)
    trainer = B1Trainer([(k, v) for k, v in model.named_parameters() if v.requires_grad],
                        TaskForward(st["serializer"], NativeForward(model, "cuda"), images, grad=True,
                                    excluded_ids=st["excluded_ids"]),
                        schedule, nl, nl_q, nb, nb_q)
    torch.cuda.reset_peak_memory_stats()
    requested = int(p["max_updates"])
    if trainer.total_updates != requested:
        raise RuntimeContractError(f"profile schedule has {trainer.total_updates} updates, requested {requested}")
    # Reserve the declared conservative first-update time until a real update has been measured;
    # the hard bound is the Slurm --time limit of the reserved job.
    reserve = 60.0 * float(p["first_update_reserve_minutes"])
    stats, stop_reason = [], None
    for u in range(requested):
        if not budget.allows(reserve):
            stop_reason = "time cap"
            break
        torch.cuda.synchronize()
        stats.append(trainer.run_update(u))
        torch.cuda.synchronize()
        reserve = max(s["seconds"] for s in stats)
    status = "complete" if len(stats) == requested else "incomplete"
    out = ctx.out_root / f"profile_{p['scope']}_{ctx.launch_sha256[:12]}"
    report = {**ctx.provenance(), "scope": p["scope"], "status": status, "requested_updates": requested,
              "completed_updates": len(stats), "stop_reason": stop_reason, "updates": stats, "lora": lora_inv,
              "device": st["device"], "peak_memory_allocated": torch.cuda.max_memory_allocated(),
              "peak_memory_reserved": torch.cuda.max_memory_reserved(),
              "note": ("SYNTHETIC images/text: does not establish real-image throughput. " if p["scope"] == "synthetic"
                       else "train-split only. ") + "Complete-update timings for horizon planning; LR horizon is not "
                      "the real one; adapters are discarded (no checkpoint)."}
    write_once_json(out / "profile.json", report)
    return {"status": status, "report": str(out / "profile.json"), "requested_updates": requested,
            "completed_updates": len(stats), "stop_reason": stop_reason}


# ------------------------------------------------------------------ evaluation
def _peft_adapter_view(tensors: Mapping) -> dict:
    """Normalize PEFT parameter names / saved keys to one form (drop the adapter-name segment)."""
    return {k.replace(".default.", "."): v for k, v in tensors.items()}


# This is a single audited transfer lineage, not permission to reuse arbitrary exports.
_EGO_TRANSFER_SEAL_SHA256 = "69149611760b02d3e3b9cb857b6d0711b0d17fd78e22ad3829e970147b0e400f"
_EGO_PARENT_SEAL_SHA256 = "facddd4dec4e608bd4b81ff11a7fbcc754a914b58a224ffa411a5aee7c599456"
_EGO_EXPORT_REPORT_SHA256 = "48ac1452c95dcda7d65078a6633796c6a7b2f1a4e7ebf48d1f8ba5c13f6df0e1"
_EGO_PINS_SHA256 = "ce23477b896b681c421c8edb5a3addfc524e8558d00942c51d7326e2d32c039a"


def _verify_export_lineage(ctx: Ctx, report: Mapping, report_path: Path) -> dict:
    """Retain same-seal checks; allow only the byte-bound, audited Ego transfer."""
    provenance = report["provenance"]
    if provenance["pins_sha256"] != ctx.pins_sha256:
        raise ExportError("export was produced under different runtime pins")
    seal = ctx.launch["seal_a"]
    if provenance["seal_a_sha256"] == seal["sha256"]:
        return {"kind": "same_seal", "seal_a_sha256": seal["sha256"]}
    parent = (ctx.launch.get("ego") or {}).get("adapter_parent_seal_a") or {}
    if not (ctx.phase == "ego_eval" and ctx.launch.get("phase") == "ego_eval"
            and ctx.launch.get("arm") == "B1"
            and ctx.pins_sha256 == _EGO_PINS_SHA256
            and seal["sha256"] == _EGO_TRANSFER_SEAL_SHA256
            and provenance["seal_a_sha256"] == _EGO_PARENT_SEAL_SHA256
            and parent.get("sha256") == _EGO_PARENT_SEAL_SHA256):
        raise ExportError("export has no verified Seal A transfer lineage")
    try:
        if sha256_file(Path(seal["path"])) != _EGO_TRANSFER_SEAL_SHA256:
            raise ExportError("Ego transfer addendum bytes differ")
        if sha256_file(Path(parent["path"])) != _EGO_PARENT_SEAL_SHA256:
            raise ExportError("Ego adapter parent Seal A bytes differ")
        addendum = Path(seal["path"]).read_text(encoding="utf-8")
    except (OSError, KeyError, TypeError) as exc:
        raise ExportError("Ego transfer requires both immutable seal files") from exc
    if (f"linked to immutable baseline Seal A, SHA256 {_EGO_PARENT_SEAL_SHA256}." not in addendum
            or f"Bound runtime pins: {_EGO_PINS_SHA256}." not in addendum):
        raise ExportError("Ego addendum does not explicitly bind parent seal and pins")
    # Exact report bytes bind selected seed17/e3/update261, training identity,
    # adapter digest, original seal, and unchanged model/runtime pins together.
    if sha256_file(report_path) != _EGO_EXPORT_REPORT_SHA256:
        raise ExportError("Ego transfer export report differs from the audited checkpoint")
    return {"kind": "ego_transfer", "parent_seal_a_sha256": _EGO_PARENT_SEAL_SHA256,
            "seal_a_sha256": _EGO_TRANSFER_SEAL_SHA256,
            "export_report_sha256": _EGO_EXPORT_REPORT_SHA256}


def _load_verified_adapter(ctx: Ctx, base, adapter_root: Path):
    from peft import PeftModel

    from .export import assert_adapter_equal, assert_native_source
    from .tensorio import load_tensors

    report = read_json(adapter_root / "export_report.json", verify=True)
    if report.get("passed") is not True:
        raise ExportError(f"{adapter_root}: export verification did not pass")
    lineage = _verify_export_lineage(ctx, report, adapter_root / "export_report.json")
    if report["identity"]["seed"] != ctx.launch["seed"]:
        raise ExportError("export seed differs from the launch seed")
    if sha256_file(adapter_root / "adapter" / "adapter_model.safetensors") != report["adapter_file_sha256"]:
        raise ExportError("adapter file differs from the verified export")
    saved, _ = load_tensors(adapter_root / "adapter" / "adapter_model.safetensors")
    model = PeftModel.from_pretrained(base, str(adapter_root / "adapter"), is_trainable=False)
    model.eval()
    eq = assert_adapter_equal(_peft_adapter_view(saved),
                              _peft_adapter_view({n: p for n, p in model.named_parameters() if ".lora_" in n}))
    return model, {"export_report_sha256": sha256_file(adapter_root / "export_report.json"), "adapter_equal": eq,
                   "native_source": assert_native_source(model), "identity": report.get("identity"),
                   "export_lineage": lineage}


def evaluate(launch_path: str) -> dict:
    launch, _ = load_launch(launch_path)
    phase = launch.get("phase")
    if phase not in ("b0_eval", "b1_eval"):
        raise RuntimeContractError(f"evaluate runs b0_eval/b1_eval launches, not {phase!r}")
    ctx = setup(launch_path, phase)
    import math

    from ..data.normbank import retention_subset
    from .data_io import normbank_input, normlens_input, split_rows
    from .evaluate import ResumableNativeEval
    from .export import assert_native_source
    from .model import NativeForward, TaskForward
    from .train import Budget

    budget = Budget(deadline_seconds(ctx.launch))  # started before data/model load: load time counts
    e = ctx.launch["eval"]
    reserve_min = e.get("first_forward_reserve_minutes")
    if not (isinstance(reserve_min, (int, float)) and not isinstance(reserve_min, bool)
            and math.isfinite(reserve_min) and 0 < 60.0 * reserve_min < budget.limit):
        raise RuntimeContractError("eval.first_forward_reserve_minutes must be an explicit finite positive value "
                                   "inside the usable time (conservative reserve before the first forward)")
    task, split = e["task"], e["split"]
    data = _data(ctx, normlens=task == "normlens", normbank=task == "normbank_retention")
    shuffle_map = None
    if task == "normlens":
        cond = e["condition"]
        records = [normlens_input(r) for r in split_rows(data["normlens"], split)]
        letters, fwd_task = NORMLENS_LETTERS, {"image": "normlens", "text_only": "normlens_text_only",
                                               "shuffle": "normlens_shuffle"}[cond]
        if cond == "shuffle":
            verify_sidecar(e["shuffle_map"])
            shuffle_map = read_json(e["shuffle_map"])
            if set(shuffle_map) != {r["example_id"] for r in records}:
                raise RuntimeContractError("shuffle map does not cover exactly the evaluated split")
    else:
        by_id = {r["example_id"]: r for r in data["normbank"]}
        records = [normbank_input(by_id[i]) for i in retention_subset(data["normbank"])]
        letters, fwd_task, cond = NORMBANK_LETTERS, "normbank", "normbank_retention"
    arm = ctx.launch["arm"]
    identity = eval_identity(ctx, task=task, split=split, condition=cond, ids=[r["example_id"] for r in records],
                             letters=letters)
    out = ctx.out_root / f"eval_{arm}_{task}_{split}_{cond}_{_json_sha(identity)[:16]}"
    st = _stack(ctx)
    model, extra = st["model"], {}
    if arm == "B1":
        model, extra = _load_verified_adapter(ctx, model, Path(e["adapter_dir"]))
    else:
        extra = {"native_source": assert_native_source(model)}
    model.eval()
    fwd = TaskForward(st["serializer"], NativeForward(model, "cuda"), data.get("images"), grad=False,
                      excluded_ids=st["excluded_ids"], shuffle_map=shuffle_map)
    run = ResumableNativeEval(out, identity, records, fwd.with_meta(fwd_task), letters, seed=ctx.launch["seed"],
                              condition=cond, task=task, arm=arm, budget=budget,
                              first_forward_reserve_s=60.0 * float(reserve_min))
    run.open()
    res = run.run()
    if res["status"] == "paused":
        return {**res, "out": str(out)}
    manifest = run.finalize({**ctx.provenance(), "device": st["device"], "determinism": st["determinism"],
                             "model": st["model_report"], "processor": st["processor_report"], "adapter": extra})
    return {"status": "complete" if manifest["complete"] else "incomplete", "out": str(out),
            "predicted": manifest["predicted"], "failed": manifest["failed"]}


def eval_identity(ctx: Ctx, *, task: str, split: str, condition: str, ids, letters) -> dict:
    """Stable scientific identity of a native evaluation (excludes launch/reservation/timing): a new
    reservation for the same evaluation resumes the same directory; any scientific change is a new run."""
    e, d = ctx.launch["eval"], ctx.launch.get("data") or {}
    adapter = None
    if ctx.launch["arm"] == "B1":
        adapter = sha256_file(Path(e["adapter_dir"]) / "export_report.json")
    return {"runtime_version": RUNTIME_VERSION, "phase": ctx.phase, "arm": ctx.launch["arm"],
            "seed": ctx.launch.get("seed"), "task": task, "split": split, "condition": condition,
            "pins_sha256": ctx.pins_sha256, "seal_a_sha256": (ctx.launch.get("seal_a") or {}).get("sha256"),
            "data_manifest_sha256": d.get("normlens_manifest_sha256" if task == "normlens" else
                                          "normbank_manifest_sha256"),
            "shuffle_map_sha256": e.get("shuffle_map_sha256") if condition == "shuffle" else None,
            "adapter_export_report_sha256": adapter, "answer_letters": list(letters),
            "expected_ids_sha256": _json_sha(list(ids))}


# ------------------------------------------------------------------ export
def export_verify(launch_path: str) -> dict:
    ctx = setup(launch_path, "export_verify")
    import torch

    from .checkpoint import load_checkpoint
    from .data_io import normlens_input, split_rows
    from .evaluate import prediction_row
    from .export import assert_adapter_equal, assert_native_source, native_repeat_floor, verify_reloads
    from .lora import adapter_sha256, adapter_tensors, apply_lora, load_adapter_tensors
    from .model import NativeForward, TaskForward, load_model
    from .tensorio import load_tensors

    ckpt_dir = Path(ctx.launch["export"]["checkpoint_dir"])
    run_dir = ckpt_dir.parent.parent
    if not str(run_dir.resolve()).startswith(str(ctx.out_root.resolve()) + os.sep):
        raise CheckpointError("checkpoint is outside this launch's output_root (untrusted)")
    selection = read_json(run_dir / "selection.json", verify=True)
    if Path(selection["checkpoint"]["dir"]).resolve() != ckpt_dir.resolve():
        raise CheckpointError("export checkpoint is not the dev-selected checkpoint of its run")
    ck = load_checkpoint(ckpt_dir, trusted_root=run_dir)
    identity = ck["state"]["identity"]
    if identity["seal_a_sha256"] != ctx.launch["seal_a"]["sha256"] or identity["pins_sha256"] != ctx.pins_sha256 or \
            identity["seed"] != ctx.launch["seed"]:
        raise CheckpointError("checkpoint was trained under a different Seal A, pins or seed")
    data = _data(ctx, normbank=False)
    dev = split_rows(data["normlens"], "dev")
    fixtures = sorted((normlens_input(r) for r in dev),
                      key=lambda r: sort_key_sha256(EXPORT_FIXTURE_PREFIX, r["example_id"]))[:POTENCY_FLOOR_INPUTS]
    fixture_ids = [r["example_id"] for r in fixtures]
    if len(fixture_ids) != POTENCY_FLOOR_INPUTS:
        raise ExportError(f"need {POTENCY_FLOOR_INPUTS} dev fixtures, found {len(fixture_ids)}")
    n_answers = len(NORMLENS_LETTERS)

    st = _stack(ctx)
    model, _ = apply_lora(st["model"], identity["seed"])
    load_adapter_tensors(model, ck["adapter"])
    if adapter_sha256(adapter_tensors(model)) != ck["state"]["adapter_sha256"]:
        raise ExportError("adapter loaded into the model differs from the checkpoint")
    model.eval()
    serializer, images, excluded = st["serializer"], data["images"], st["excluded_ids"]

    def fixture_fn(m):
        fn = TaskForward(serializer, NativeForward(m, "cuda"), images, grad=False,
                         excluded_ids=excluded).with_meta("normlens")
        out = {}
        for rec in fixtures:
            logits, ids, meta = fn(rec)
            r = prediction_row(rec["example_id"], logits, ids, NORMLENS_LETTERS, seed=identity["seed"],
                               condition="image", task="normlens", arm="B1", meta=meta)
            out[rec["example_id"]] = {"full_logprobs": r["full_logprobs"], "argmax_token_id": r["argmax_token_id"]}
        return out

    pre_source = assert_native_source(model)
    floor, runs = native_repeat_floor(fixture_fn, model, POTENCY_FLOOR_REPEATS, fixture_ids, n_answers)
    reference = runs[0]
    export_dir = ctx.out_root / f"export_{run_dir.name}_{ckpt_dir.name}"
    if export_dir.exists():
        raise ExportError(f"{export_dir} exists (exports are write-once)")
    model.save_pretrained(str(export_dir / "adapter"), safe_serialization=True)
    st["processor"].save_pretrained(str(export_dir / "processor"))
    saved, _ = load_tensors(export_dir / "adapter" / "adapter_model.safetensors")
    assert_adapter_equal(_peft_adapter_view(saved), _peft_adapter_view(adapter_tensors(model)))
    del model, st
    torch.cuda.empty_cache()
    reload_sources = []

    def reload_once():
        from peft import PeftModel

        base, _ = load_model(ctx.model_dir, ctx.pins, device="cuda")
        m = PeftModel.from_pretrained(base, str(export_dir / "adapter"), is_trainable=False)
        m.eval()
        reload_sources.append(assert_native_source(m))
        assert_adapter_equal(_peft_adapter_view(saved),
                             _peft_adapter_view({n: p for n, p in m.named_parameters() if ".lora_" in n}))
        result = fixture_fn(m)
        del m, base
        torch.cuda.empty_cache()
        return result

    report = verify_reloads(reference, floor, [reload_once, reload_once], fixture_ids, n_answers)
    write_once_json(export_dir / "export_report.json", {
        "passed": report["passed"], "verification": report, "fixtures": fixture_ids, "reference": reference,
        "identity": identity, "checkpoint": str(ckpt_dir), "adapter_sha256": ck["state"]["adapter_sha256"],
        "adapter_file_sha256": sha256_file(export_dir / "adapter" / "adapter_model.safetensors"),
        "native_source_pre_save": pre_source, "native_source_reloads": reload_sources,
        "provenance": ctx.provenance()})
    return {"status": "complete", "export_dir": str(export_dir), "verification": report}
