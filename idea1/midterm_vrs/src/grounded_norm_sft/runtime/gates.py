"""Launch gates (plan 7, 9, 13.1, A1, A6). Evaluated on metadata only, before any model load.

Every GPU/model phase needs a per-job launch file. Missing, placeholder or nonfinite values fail;
nothing defaults to approval. Problems are collected and reported together, each prefixed with
its gate name (e.g. ``GATE approval: ...``).

Authorization: the launch references ONE user approval envelope (path + SHA256, see ledger.py)
that enumerates the phase, and a ledger reservation made by the launcher for this exact attempt.
The runtime only verifies them (and marks the reservation used at job start).

Phases and their gates:
  gpu_profile_preseal  approval, reservation, job, pins, profile scope (+ rights/train data if
                       train_only); Seal A NOT required: synthetic or train-split-only profiling.
  b0_eval / b1_eval    approval, reservation, job, pins, rights, Seal A, data, eval spec (+ export)
  b1_train             approval, reservation, job, pins, rights, Seal A, data, precommitted
                       horizon/seed, comparable-checkpoint policy, dev-time reserve
  export_verify        approval, reservation, job, pins, rights, Seal A, NormLens data, checkpoint
CPU synthetic tests and processor-only verification need none of these.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from ..constants import (ALLOCATION, ALLOWED_EPOCHS, DIAGNOSTIC_ONLY_ARMS, INTERVENTION_ARMS, PAPER_SEEDS,
                         REMOTE_ROOT)
from ..hashing import sha256_file
from .errors import ArmRefused, GateError
from .ledger import GRES, PARTITION, SU_EPS, check_reservation, finite_positive, is_v2, is_v3, job_su, load_approval
from .pins import open_pin_gates

LAUNCH_SCHEMA = "gn_baseline_launch_v1"
CHECKPOINT_POLICY = "epoch_boundary_dev_cond_nll_ties_earlier"
INVENTORY_PHASE = "gpu_inventory"
MEDIA_PHASE = "media_decode"  # CPU decode on the compute node of an approved one-H100 reservation; no model
EGO_PHASES = ("ego_profile_preseal", "ego_eval")
MODEL_PHASES = ("gpu_profile_preseal", "b0_eval", "b1_train", "b1_eval", "export_verify") + EGO_PHASES
PHASES = (INVENTORY_PHASE, MEDIA_PHASE) + MODEL_PHASES
PHASE_ARM = {INVENTORY_PHASE: None, MEDIA_PHASE: None, "gpu_profile_preseal": "B1", "b0_eval": "B0",
             "b1_train": "B1", "b1_eval": "B1", "export_verify": "B1", "ego_profile_preseal": "B0",
             "ego_eval": ("B0", "B1")}
# Ego cohorts (provenance/egonormia_eval_ids_v1.json fields). profile5: pre-seal EOS/length/runtime
# profile, never accuracy; full400: sealed primary transfer; diag32: A5.1 rotations + candidates
# only, paired with a completed full400 primary run (primary generation is never re-run).
EGO_COHORTS = {"ego_profile_preseal": ("profile5",), "ego_eval": ("full400", "diag32")}
EGO_MAX_PROMPT_RANGE = (1024, 8192)
EVAL_TASKS = ("normlens", "normbank_retention")
NORMLENS_CONDITIONS = ("image", "text_only", "shuffle")
_PLACEHOLDERS = {"todo", "tbd", "placeholder", "changeme", "xxx", "fill_me", "fixme", "none", "null", "n/a", "na",
                 "unknown", "pending", "?"}


def check_arm(arm: str) -> str:
    if arm in ("B0", "B1"):
        return arm
    if arm in INTERVENTION_ARMS:
        raise ArmRefused([f"GATE arm: {arm} is an intervention arm; refused until the reviewed SRER integration "
                          "and Seal B exist (this runtime slice is baseline-only)"])
    if arm in DIAGNOSTIC_ONLY_ARMS:
        raise ArmRefused([f"GATE arm: {arm} training is stopped under plan 10.2 until a separate amendment"])
    if arm == "B0-cal":
        raise ArmRefused(["GATE arm: B0-cal has no forward; use grounded_norm_sft.metrics.calibration on B0 dev rows"])
    raise ArmRefused([f"GATE arm: unknown arm {arm!r}"])


def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        v = value.strip()
        return not v or v.lower() in _PLACEHOLDERS or bool(re.search(r"<[^>]*>", v)) or v.upper().startswith("TODO")
    return False


def _parse_time(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        t = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _finite_int(v: Any) -> bool:
    return finite_positive(v) and float(v).is_integer()


class _Collector:
    def __init__(self):
        self.problems: list[str] = []

    def add(self, gate: str, msg: str) -> None:
        self.problems.append(f"GATE {gate}: {msg}")

    def section(self, launch: Mapping, gate: str, key: str | None = None) -> dict:
        sec = launch.get(key or gate)
        if not isinstance(sec, Mapping):
            self.add(gate, f"section {key or gate!r} missing")
            return {}
        return dict(sec)

    def real(self, sec: Mapping, gate: str, field: str) -> Any:
        v = sec.get(field)
        if is_placeholder(v):
            self.add(gate, f"{field} missing or placeholder")
            return None
        return v

    def timestamp(self, sec: Mapping, gate: str, field: str, now: _dt.datetime) -> None:
        t = _parse_time(sec.get(field))
        if t is None:
            self.add(gate, f"{field} must be an ISO-8601 timestamp with timezone")
        elif t > now + _dt.timedelta(minutes=5):
            self.add(gate, f"{field} is in the future")


def load_launch(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    if not path.exists():
        raise GateError([f"GATE launch: launch file {path} does not exist"])
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def _check_file_hash(c: _Collector, gate: str, path: Any, digest: Any, check_files: bool) -> None:
    if is_placeholder(path) or is_placeholder(digest):
        c.add(gate, "path and sha256 are both required")
        return
    if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        c.add(gate, f"sha256 for {path} is not 64 lowercase hex")
        return
    if check_files:
        p = Path(path)
        if not p.exists():
            c.add(gate, f"{path} does not exist")
        elif sha256_file(p) != digest:
            c.add(gate, f"{path} sha256 does not match the launch file")


def _authorization(c: _Collector, launch: Mapping, phase: str, pins: Mapping, check_files: bool,
                   require_reservation: bool) -> dict | None:
    """Returns the validated approval envelope (or None)."""
    a = c.section(launch, "authorization")
    if not a:
        return None
    _check_file_hash(c, "approval", a.get("approval_path"), a.get("approval_sha256"), check_files)
    if not check_files or any(p.startswith("GATE approval") for p in c.problems):
        return None
    try:
        env = load_approval(a["approval_path"], a["approval_sha256"], PHASES, pins.get("psc", {}).get("approved_su_budget"))
    except GateError as exc:
        c.problems.extend(exc.problems)
        return None
    if is_v3(env) and require_reservation and launch.get("queue_policy") != "held_afterok_root_release":
        c.add("authorization", "v3 launches require queue_policy held_afterok_root_release")
    spec = env["phases"].get(phase)
    if spec is None:
        c.add("approval", f"phase {phase!r} is not enumerated in approval envelope {env['approval_id']}")
        return env
    tl = (launch.get("job") or {}).get("time_limit_minutes")
    if finite_positive(tl):
        if tl > spec["max_wall_minutes"]:
            c.add("job", f"time_limit_minutes {tl} exceeds the phase's approved max_wall_minutes {spec['max_wall_minutes']}")
        if job_su(env, tl) > spec["max_su"] + SU_EPS:
            c.add("job", f"time cap costs {job_su(env, tl):.2f} SU > phase max_su {spec['max_su']}")
    if require_reservation:
        c.problems.extend(check_reservation(env, a["approval_sha256"], a.get("reservation_id"), phase, tl, a.get("after_job")))
    return env


def _job(c: _Collector, launch: Mapping, pins: Mapping, phase: str) -> None:
    j = c.section(launch, "job")
    if not j:
        return
    psc = pins.get("psc", {})
    for field, want, pinned in (("account", ALLOCATION, psc.get("account")), ("partition", PARTITION, psc.get("partition")),
                                ("gres", GRES, psc.get("gres")), ("remote_root", REMOTE_ROOT, psc.get("remote_root"))):
        if j.get(field) != want or pinned != want:
            c.add("job", f"{field} must be exactly {want!r} (launch {j.get(field)!r}, pins {pinned!r})")
    tl, m = j.get("time_limit_minutes"), j.get("stop_margin_minutes")
    if not _finite_int(tl):
        c.add("job", "time_limit_minutes must be an explicit finite positive integer")
    if not finite_positive(m) or (finite_positive(tl) and m >= tl):
        c.add("job", "stop_margin_minutes must be finite, positive and below time_limit_minutes")
    if phase == "b1_train":
        d = j.get("dev_reserve_minutes")
        if not finite_positive(d) or (finite_positive(tl) and finite_positive(m) and d >= tl - m):
            c.add("job", "dev_reserve_minutes must be finite, positive and fit inside the time cap")
    paths = launch.get("paths") if isinstance(launch.get("paths"), Mapping) else {}
    out = paths.get("output_root")
    if is_placeholder(out) or not str(Path(str(out))).startswith(REMOTE_ROOT + "/"):
        c.add("paths", f"output_root must be inside {REMOTE_ROOT}/")
    if phase not in (INVENTORY_PHASE, MEDIA_PHASE) and is_placeholder(paths.get("model_dir")):
        c.add("paths", "model_dir missing (local pinned snapshot; never downloaded)")


def _rights(c: _Collector, launch: Mapping, datasets: list[str], now: _dt.datetime) -> None:
    r = c.section(launch, "rights", "rights_attestation")
    if not r:
        return
    for f in ("attested_by", "statement"):
        c.real(r, "rights", f)
    c.timestamp(r, "rights", "attested_at", now)
    missing = [d for d in datasets if d not in set(r.get("datasets") or [])]
    if missing:
        c.add("rights", f"attestation does not cover {missing} (user attestation is recorded, not certified here)")


def _data(c: _Collector, launch: Mapping, need_normlens: bool, need_normbank: bool, check_files: bool) -> dict:
    d = c.section(launch, "data")
    if not d:
        return {}
    hashes = {}
    if need_normlens:
        _check_file_hash(c, "data", d.get("normlens_manifest"), d.get("normlens_manifest_sha256"), check_files)
        hashes["normlens"] = d.get("normlens_manifest_sha256")
        if is_placeholder(d.get("normlens_image_dir")):
            c.add("data", "normlens_image_dir missing")
        elif check_files and not Path(d["normlens_image_dir"]).is_dir():
            c.add("data", f"normlens_image_dir {d['normlens_image_dir']} does not exist")
    if need_normbank:
        _check_file_hash(c, "data", d.get("normbank_manifest"), d.get("normbank_manifest_sha256"), check_files)
        hashes["normbank"] = d.get("normbank_manifest_sha256")
    return hashes


def _seal_a(c: _Collector, launch: Mapping, must_contain: dict[str, str], check_files: bool) -> None:
    s = c.section(launch, "seal_a")
    if not s:
        return
    _check_file_hash(c, "seal_a", s.get("path"), s.get("sha256"), check_files)
    if check_files and s.get("path") and Path(s["path"]).exists():
        text = Path(s["path"]).read_text(encoding="utf-8", errors="replace")
        for label, digest in must_contain.items():
            if digest and digest not in text:
                c.add("seal_a", f"Seal A does not record the {label} sha256 {digest}")


def _seed(c: _Collector, launch: Mapping, allow_null: bool = False) -> None:
    s = launch.get("seed")
    if allow_null:
        if s is not None:
            c.add("seed", "B0 is the frozen model; seed must be null")
    elif s not in PAPER_SEEDS or isinstance(s, bool):
        c.add("seed", f"seed must be one of {PAPER_SEEDS}")


def _horizon(c: _Collector, launch: Mapping, now: _dt.datetime) -> None:
    h = c.section(launch, "horizon")
    if h:
        if h.get("epochs") not in ALLOWED_EPOCHS or isinstance(h.get("epochs"), bool):
            c.add("horizon", f"epochs must be precommitted as one of {ALLOWED_EPOCHS}")
        c.timestamp(h, "horizon", "committed_at", now)
        c.real(h, "horizon", "basis")
    if launch.get("checkpoint_policy") != CHECKPOINT_POLICY:
        c.add("checkpoint_policy", f"must be {CHECKPOINT_POLICY!r} (identical across compared arms)")
    _seed(c, launch)


def _eval_spec(c: _Collector, launch: Mapping, phase: str, check_files: bool) -> dict:
    e = c.section(launch, "eval")
    if not e:
        return {}
    if e.get("split") not in ("dev", "test"):
        c.add("eval", "split must be 'dev' or 'test'")
    if e.get("task") not in EVAL_TASKS:
        c.add("eval", f"task must be one of {EVAL_TASKS}")
    if e.get("task") == "normlens" and e.get("condition") not in NORMLENS_CONDITIONS:
        c.add("eval", f"condition must be one of {NORMLENS_CONDITIONS}")
    if e.get("task") == "normbank_retention" and e.get("split") != "test":
        c.add("eval", "NormBank retention uses the official-test retention subset (split 'test')")
    if e.get("condition") == "shuffle":
        _check_file_hash(c, "eval", e.get("shuffle_map"), e.get("shuffle_map_sha256"), check_files)
    if phase == "b1_eval":
        a = e.get("adapter_dir")
        if is_placeholder(a):
            c.add("eval", "adapter_dir (verified native export) missing")
        elif check_files and not (Path(a) / "export_report.json").exists():
            c.add("export", f"{a} has no export_report.json; B1 evaluation requires a verified native export")
    _seed(c, launch, allow_null=phase == "b0_eval")
    return e


def check_environment(launch: Mapping, env: Mapping[str, str]) -> list[str]:
    """Inside the authorized Slurm allocation? (GPU model/type is re-checked on device before load.)"""
    j = launch.get("job") if isinstance(launch.get("job"), Mapping) else {}
    problems = []
    if not env.get("SLURM_JOB_ID"):
        problems.append("GATE environment: not inside a Slurm job (SLURM_JOB_ID unset)")
    if env.get("SLURM_JOB_ACCOUNT") != j.get("account", ALLOCATION):
        problems.append(f"GATE environment: SLURM_JOB_ACCOUNT={env.get('SLURM_JOB_ACCOUNT')!r} is not {ALLOCATION!r}")
    if env.get("SLURM_JOB_PARTITION") != PARTITION:
        problems.append(f"GATE environment: SLURM_JOB_PARTITION={env.get('SLURM_JOB_PARTITION')!r} is not {PARTITION!r}")
    if env.get("SLURM_GPUS_ON_NODE") not in (None, "1"):
        problems.append(f"GATE environment: SLURM_GPUS_ON_NODE={env.get('SLURM_GPUS_ON_NODE')!r}, expected 1")
    return problems


def check_launch(launch: Mapping, phase: str, pins: Mapping, pins_sha256: str, env: Mapping[str, str] | None,
                 check_files: bool = True, now: _dt.datetime | None = None, require_reservation: bool = True) -> dict:
    """Raise GateError listing every missing/invalid gate; return a gate report otherwise."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    c = _Collector()
    code_root = (launch.get("paths") or {}).get("code_root")
    snapshot_sha = (launch.get("authorization") or {}).get("source_snapshot_sha256")
    manifest_sha = (launch.get("authorization") or {}).get("source_manifest_sha256")
    if code_root is not None or snapshot_sha is not None or manifest_sha is not None:
        from .source_snapshot import verify_source_snapshot
        if not isinstance(code_root, str) or not isinstance(snapshot_sha, str) or not isinstance(manifest_sha, str):
            c.add("source snapshot", "code-root and snapshot SHA256 must both be bound")
        else:
            try:
                verify_source_snapshot(code_root, snapshot_sha)
                if sha256_file(Path(code_root) / "SHA256SUMS") != manifest_sha:
                    c.add("source snapshot", "manifest SHA256 differs from bound launch")
                if env is not None and Path(__file__).resolve().parents[3] != Path(code_root):
                    c.add("source snapshot", "runtime was not imported from the bound code-root")
            except (GateError, OSError) as exc:
                c.problems.extend(exc.problems if isinstance(exc, GateError) else [f"GATE source snapshot: {exc}"])
    if launch.get("schema") != LAUNCH_SCHEMA:
        c.add("launch", f"schema must be {LAUNCH_SCHEMA!r}")
    if phase not in PHASES:
        raise GateError([f"GATE phase: unknown phase {phase!r}; expected one of {PHASES}"])
    if launch.get("phase") != phase:
        c.add("phase", f"launch file is for phase {launch.get('phase')!r}, command requires {phase!r}")
    if phase == INVENTORY_PHASE:
        return _check_inventory(c, launch, pins, pins_sha256, env, check_files, now, require_reservation)
    if phase == MEDIA_PHASE:
        return _check_media(c, launch, pins, pins_sha256, env, check_files, now, require_reservation)
    arm = launch.get("arm")
    check_arm(arm)
    allowed = PHASE_ARM[phase] if isinstance(PHASE_ARM[phase], tuple) else (PHASE_ARM[phase],)
    if arm not in allowed:
        c.add("arm", f"phase {phase} runs arm(s) {allowed}, launch says {arm!r}")
    _job(c, launch, pins, phase)
    approval = _authorization(c, launch, phase, pins, check_files, require_reservation)
    for g in open_pin_gates(pins):
        # A validated v2 envelope is itself the budget authority (bound to the user's approval receipt),
        # so the unset pins psc.approved_su_budget is not an open gate for it. Every other pin stays required.
        if g == "psc.approved_su_budget" and approval is not None and (is_v2(approval) or is_v3(approval)):
            continue
        c.add("pins", f"{g} is null (open gate in configs/runtime_pins.json)")
    if env is not None:
        c.problems.extend(check_environment(launch, env))

    if phase == "gpu_profile_preseal":
        p = c.section(launch, "profile")
        scope = p.get("scope")
        if scope not in ("synthetic", "train_only"):
            c.add("profile", "scope must be 'synthetic' or 'train_only' (pre-seal profiling never reads dev/test)")
        if not _finite_int(p.get("max_updates")):
            c.add("profile", "max_updates must be an explicit finite positive integer")
        r = p.get("first_update_reserve_minutes")
        tl, m = (launch.get("job") or {}).get("time_limit_minutes"), (launch.get("job") or {}).get("stop_margin_minutes")
        if not finite_positive(r) or (finite_positive(tl) and finite_positive(m) and r >= tl - m):
            c.add("profile", "first_update_reserve_minutes (conservative wall time reserved before the first, "
                             "unmeasured update) must be finite, positive and fit inside the time cap")
        _seed(c, launch)
        if scope == "train_only":
            _rights(c, launch, ["normlens", "normbank"], now)
            _data(c, launch, True, True, check_files)
    elif phase in EGO_PHASES:
        _ego_spec(c, launch, phase, arm, pins_sha256, check_files, now)
    else:
        if phase in ("b0_eval", "b1_eval"):
            e = _eval_spec(c, launch, phase, check_files)
            need_nl, need_nb = e.get("task") == "normlens", e.get("task") == "normbank_retention"
        else:
            need_nl, need_nb = True, phase == "b1_train"
        _rights(c, launch, [d for d, need in (("normlens", need_nl), ("normbank", need_nb)) if need], now)
        hashes = _data(c, launch, need_nl, need_nb, check_files)
        _seal_a(c, launch, {"runtime pins": pins_sha256, **{f"{k} manifest": v for k, v in hashes.items()}},
                check_files)
        if phase == "b1_train":
            _horizon(c, launch, now)
        if phase == "export_verify":
            x = c.section(launch, "export")
            if is_placeholder(x.get("checkpoint_dir")):
                c.add("export", "checkpoint_dir missing")
            _seed(c, launch)
    if c.problems:
        raise GateError(c.problems)
    return {"phase": phase, "arm": arm, "checked_at": now.isoformat(), "pins_sha256": pins_sha256,
            "environment_checked": env is not None, "files_checked": check_files,
            "reservation_checked": require_reservation and check_files}


def _ego_spec(c: _Collector, launch: Mapping, phase: str, arm: str, pins_sha256: str, check_files: bool,
              now: _dt.datetime) -> None:
    """Ego gates. Both phases load the model, so all model pins/approval/reservation/job gates above
    apply. Pre-seal profile: B0, first-5 profiling IDs only, no Seal A, no accuracy. Final eval:
    Seal A recording pins + Ego source/ID/sheet manifests; B1 additionally a verified native export."""
    from ..constants import EGO_SOURCE_SHA256
    from ..io_utils import read_json

    e = c.section(launch, "ego")
    if not e:
        return
    if e.get("diagnostic_protocol") is not None:
        from .ego_generation_diagnostic import PROTOCOL, validate_protocol
        from .errors import RuntimeContractError
        if phase != "ego_eval" or e.get("cohort") != "diag32" or e.get("diagnostic_protocol") != PROTOCOL:
            c.add("ego", "generation protocol requires explicit ego_eval/diag32 v1 diagnostic scope")
        if not finite_positive(launch.get("job", {}).get("time_limit_minutes")) or launch["job"]["time_limit_minutes"] > 360:
            c.add("ego", "generation diagnostic is bounded to at most six hours")
        _check_file_hash(c, "ego", e.get("protocol_manifest_path"), e.get("protocol_manifest_sha256"), check_files)
        if check_files and Path(e.get("protocol_manifest_path") or "").is_file():
            try:
                validate_protocol(read_json(e["protocol_manifest_path"], verify=True))
            except (RuntimeContractError, ValueError, KeyError) as exc:
                c.add("ego", str(exc))
    _rights(c, launch, ["egonormia"], now)
    if e.get("cohort") not in EGO_COHORTS[phase]:
        c.add("ego", f"cohort must be one of {EGO_COHORTS[phase]} for {phase}")
    if e.get("source_sha256") != EGO_SOURCE_SHA256:
        c.add("ego", f"source_sha256 must be the pinned production final_data.json {EGO_SOURCE_SHA256}")
    hashes = {}
    for key in ("source", "ids_manifest", "sheet_manifest"):
        _check_file_hash(c, "ego", e.get(f"{key}_path"), e.get(f"{key}_sha256"), check_files)
        hashes[f"Ego {key}"] = e.get(f"{key}_sha256")
    mp = e.get("max_prompt_tokens")
    if not _finite_int(mp) or not EGO_MAX_PROMPT_RANGE[0] <= mp <= EGO_MAX_PROMPT_RANGE[1]:
        c.add("ego", f"max_prompt_tokens must be an explicit integer in {EGO_MAX_PROMPT_RANGE} (prompts are never "
                     "truncated; overlength items are recorded failures)")
    r = e.get("first_item_reserve_minutes")
    tl, m = (launch.get("job") or {}).get("time_limit_minutes"), (launch.get("job") or {}).get("stop_margin_minutes")
    if not finite_positive(r) or (finite_positive(tl) and finite_positive(m) and r >= tl - m):
        c.add("ego", "first_item_reserve_minutes must be finite, positive and fit inside the time cap")
    if e.get("cohort") == "diag32":
        if is_placeholder(e.get("primary_manifest_path")) or is_placeholder(e.get("primary_manifest_sha256")):
            c.add("ego", "diag32 requires primary_manifest_path/sha256 of a completed full400 run (never re-run)")
        else:
            _check_file_hash(c, "ego", e.get("primary_manifest_path"), e.get("primary_manifest_sha256"), check_files)
    elif e.get("primary_manifest_path") is not None:
        c.add("ego", "primary_manifest_path is only for the diag32 cohort")
    if phase == "ego_profile_preseal":
        _seed(c, launch, allow_null=True)
        for k in ("adapter_dir", "primary_manifest_path"):
            if e.get(k) is not None:
                c.add("ego", f"pre-seal profile must not set {k}")
        return
    _seal_a(c, launch, {"runtime pins": pins_sha256, **hashes}, check_files)
    _seed(c, launch, allow_null=arm == "B0")
    if arm == "B1":
        a = e.get("adapter_dir")
        if is_placeholder(a):
            c.add("ego", "adapter_dir (verified native export) missing for B1")
        elif check_files and not (Path(a) / "export_report.json").exists():
            c.add("export", f"{a} has no export_report.json; B1 Ego evaluation requires a verified native export")
    elif e.get("adapter_dir") is not None:
        c.add("ego", "B0 must not set adapter_dir")


INVENTORY_FORBIDDEN_SECTIONS = ("arm", "seed", "data", "seal_a", "rights_attestation", "horizon", "eval", "profile",
                                "export", "checkpoint_policy")


def _check_inventory(c: _Collector, launch: Mapping, pins: Mapping, pins_sha256: str, env, check_files: bool,
                     now: _dt.datetime, require_reservation: bool) -> dict:
    """Minimal inventory gates: approval envelope (which requires the user's psc.approved_su_budget),
    single-use reservation, exact allocation/partition/gres/time, Slurm identity. Deliberately NOT the
    model/runtime pins (gpu_driver, cuda, torch, shards, ...): this phase produces the evidence for them.
    It cannot carry model/data sections, so it cannot be repurposed as a model launch."""
    present = [k for k in INVENTORY_FORBIDDEN_SECTIONS if launch.get(k) is not None]
    if present:
        c.add("inventory", f"inventory launches must not carry model/data sections {present}")
    if (launch.get("paths") or {}).get("model_dir") is not None:
        c.add("inventory", "inventory launches must not reference model_dir")
    inv = launch.get("inventory") if isinstance(launch.get("inventory"), Mapping) else {}
    if is_placeholder(inv.get("expected_torch_version")):
        c.add("inventory", "expected_torch_version (exact installed torch build) missing")
    _job(c, launch, pins, INVENTORY_PHASE)
    _authorization(c, launch, INVENTORY_PHASE, pins, check_files, require_reservation)
    if env is not None:
        c.problems.extend(check_environment(launch, env))
    if c.problems:
        raise GateError(c.problems)
    return {"phase": INVENTORY_PHASE, "arm": None, "checked_at": now.isoformat(), "pins_sha256": pins_sha256,
            "model_pins_checked": False, "environment_checked": env is not None, "files_checked": check_files,
            "reservation_checked": require_reservation and check_files}


MEDIA_COHORTS = ("profile5", "full400")
MEDIA_DOWNLOAD_RECEIPT_SHA256 = "e3600b4e870e4e23f5b965499786da956c0adc9b14d50e000670d2fa704b6893"  # root: 400 verified
MEDIA_PYTHON = REMOTE_ROOT + "/.venv/bin/python"  # the pinned CPU decoder environment (not .venv-runtime)
MEDIA_FORBIDDEN_SECTIONS = INVENTORY_FORBIDDEN_SECTIONS + ("inventory", "ego")
DECODER_RECEIPT_SCHEMA = "decoder_wheel_provenance/v1"
DECODER_WHEELS = ("av", "pillow")


def _check_media(c: _Collector, launch: Mapping, pins: Mapping, pins_sha256: str, env, check_files: bool,
                 now: _dt.datetime, require_reservation: bool) -> dict:
    """media_decode: approval (v2 envelope), single-use reservation, exact job identity, Slurm identity,
    fixed Ego source chain (frozen 400 manifest, download receipt, plan) and verified decoder wheels.
    Deliberately NOT model pins or Seal A: decode is a prerequisite of overlap and Seal A. No model,
    data or Ego-eval sections are allowed, so it cannot be repurposed as a model launch."""
    from ..data.overlap import FROZEN_MANIFEST_SHA256

    present = [k for k in MEDIA_FORBIDDEN_SECTIONS if launch.get(k) is not None]
    if present:
        c.add("media", f"media_decode launches must not carry model/data sections {present}")
    paths = launch.get("paths") if isinstance(launch.get("paths"), Mapping) else {}
    if paths.get("model_dir") is not None:
        c.add("media", "media_decode launches must not reference model_dir")
    if paths.get("python") != MEDIA_PYTHON:
        c.add("media", f"paths.python must be the CPU decoder environment {MEDIA_PYTHON!r}")
    m = c.section(launch, "media")
    if m.get("operation") == "normlens_rgb_reaudit":
        return _check_rgb_reaudit(c, launch, pins, pins_sha256, env, check_files, now, require_reservation)
    if m.get("operation") not in (None, "egonormia_decode"):
        c.add("media", "unknown media operation")
    if m:
        cohort = m.get("cohort")
        if cohort not in MEDIA_COHORTS:
            c.add("media", f"cohort must be one of {MEDIA_COHORTS} (profile5 = frozen profiling_ids only)")
        if m.get("frozen_sha256") != FROZEN_MANIFEST_SHA256:
            c.add("media", f"frozen_sha256 must be the accepted frozen 400 manifest {FROZEN_MANIFEST_SHA256}")
        if m.get("receipt_sha256") != MEDIA_DOWNLOAD_RECEIPT_SHA256:
            c.add("media", f"receipt_sha256 must be the verified 400-item download receipt {MEDIA_DOWNLOAD_RECEIPT_SHA256}")
        for key in ("frozen", "plan", "receipt", "decoder_receipt"):
            _check_file_hash(c, "media", m.get(f"{key}_path"), m.get(f"{key}_sha256"), check_files)
        root, out = m.get("media_root"), m.get("out_dir")
        if is_placeholder(root):
            c.add("media", "media_root (verified download root) missing")
        elif check_files and not Path(root).is_dir():
            c.add("media", f"media_root {root} does not exist")
        if is_placeholder(out) or not str(out).startswith(str(paths.get("output_root") or "\0") + "/"):
            c.add("media", "out_dir must be inside paths.output_root")
        elif cohort in MEDIA_COHORTS and not Path(str(out)).name.startswith(cohort):
            c.add("media", f"out_dir name must start with the cohort {cohort!r} (profile and production never share)")
        if m.get("workers") != 1:
            c.add("media", "workers must be 1 (resumable per-item decode)")
        r = m.get("item_reserve_seconds")
        tl, mg = (launch.get("job") or {}).get("time_limit_minutes"), (launch.get("job") or {}).get("stop_margin_minutes")
        if not finite_positive(r) or (finite_positive(tl) and finite_positive(mg) and r >= 60 * (tl - mg)):
            c.add("media", "item_reserve_seconds must be finite, positive and fit inside the usable time")
        if check_files and not any(p.startswith("GATE media") for p in c.problems):
            _decoder_receipt(c, m["decoder_receipt_path"])
    _job(c, launch, pins, MEDIA_PHASE)
    _authorization(c, launch, MEDIA_PHASE, pins, check_files, require_reservation)
    if env is not None:
        c.problems.extend(check_environment(launch, env))
    if c.problems:
        raise GateError(c.problems)
    return {"phase": MEDIA_PHASE, "arm": None, "checked_at": now.isoformat(), "pins_sha256": pins_sha256,
            "model_pins_checked": False, "environment_checked": env is not None, "files_checked": check_files,
            "reservation_checked": require_reservation and check_files}


def _decoder_receipt(c: _Collector, path: str) -> None:
    try:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        c.add("media", f"decoder receipt unreadable: {exc}")
        return
    names = {str((w.get("wheel") or {}).get("name", "")).lower() for w in r.get("wheels") or [] if isinstance(w, Mapping)}
    if r.get("schema") != DECODER_RECEIPT_SCHEMA or r.get("status") != "verified":
        c.add("media", f"decoder receipt must be a {DECODER_RECEIPT_SCHEMA} receipt with status 'verified'")
    missing = [n for n in DECODER_WHEELS if n not in names]
    if missing:
        c.add("media", f"decoder receipt does not cover wheels {missing}")


def deadline_seconds(launch: Mapping) -> float:
    """Wall-clock budget for this process: job time cap minus the declared stop margin."""
    j = launch["job"]
    s = 60.0 * (float(j["time_limit_minutes"]) - float(j["stop_margin_minutes"]))
    if not math.isfinite(s) or s <= 0:
        raise GateError(["GATE job: nonpositive or nonfinite usable time"])
    return s


RGB_LEGACY_SHA256 = "1785d045f34d9ca719188492dc3f3388e2467c68cb9b230fbacfb36ed6327705"
RGB_DECODER_SHA256 = "c7a21536350b7a78376b5baa7f83dfa55cdb215bc5b6b12f872eaea1ec1f0711"


def _check_rgb_reaudit(c, launch, pins, pins_sha256, env, check_files, now, require_reservation):
    m, j = launch["media"], launch.get("job") or {}
    for key, want in (("dataset_dir", "data/raw/normlens"), ("legacy_path", "private/legacy_exact_rgb_v1.jsonl"),
                      ("legacy_sha256", RGB_LEGACY_SHA256), ("expected_images", 1742),
                      ("decoder_receipt_path", "provenance/decoder_provenance_linux_v1.json"),
                      ("decoder_receipt_sha256", RGB_DECODER_SHA256), ("workers", 1)):
        if m.get(key) != want:
            c.add("media", f"{key} must be exactly {want!r}")
    if j.get("time_limit_minutes") != 10 or j.get("stop_margin_minutes") != 2:
        c.add("media", "RGB reaudit requires a 10 minute cap and 2 minute stop margin (1/3 SU)")
    out, output_root = m.get("out_dir"), (launch.get("paths") or {}).get("output_root")
    if is_placeholder(out) or Path(str(output_root)) not in Path(str(out)).parents:
        c.add("media", "out_dir must be inside paths.output_root")
    for key in ("legacy", "decoder_receipt"):
        _check_file_hash(c, "media", m.get(f"{key}_path"), m.get(f"{key}_sha256"), check_files)
    if check_files:
        if not Path(str(m.get("dataset_dir"))).is_dir():
            c.add("media", "NormLens dataset directory missing")
        if not any(p.startswith("GATE media") for p in c.problems):
            _decoder_receipt(c, m["decoder_receipt_path"])
            try:
                rgb_decoder_identity(m["decoder_receipt_path"])
            except (ValueError, OSError, ImportError) as exc:
                c.add("media", str(exc))
    _job(c, launch, pins, MEDIA_PHASE)
    _authorization(c, launch, MEDIA_PHASE, pins, check_files, require_reservation)
    if env is not None:
        c.problems.extend(check_environment(launch, env))
    if c.problems:
        raise GateError(c.problems)
    return {"phase": MEDIA_PHASE, "operation": "normlens_rgb_reaudit", "arm": None,
            "checked_at": now.isoformat(), "pins_sha256": pins_sha256, "model_pins_checked": False,
            "environment_checked": env is not None, "files_checked": check_files,
            "reservation_checked": require_reservation and check_files}


def rgb_decoder_identity(receipt_path):
    """Match the active Pillow import/build to the immutable Linux wheel receipt."""
    import sys
    import PIL
    from ..data.images import decoder_inventory

    receipt = json.loads(Path(receipt_path).read_text())
    entry = next(w for w in receipt["wheels"] if w["wheel"]["name"].lower() == "pillow")
    facts = entry["import_facts"]
    actual = decoder_inventory()
    if (sys.platform != "linux" or PIL.__version__ != "13.0.0" or
            str(Path(PIL.__file__).resolve()) != facts["expected_module_file"] or
            actual["libjpeg_turbo"] != facts["libjpeg_turbo"]):
        raise ValueError("active Pillow identity does not match the pinned Linux Pillow 13 receipt")
    return actual
