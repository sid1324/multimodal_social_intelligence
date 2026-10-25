"""Compute approval envelope and cumulative SU ledger.

The user grants ONE bounded approval envelope (a JSON file referenced by path + SHA256 from every
launch) that enumerates permitted phases with per-phase SU and wall-time ceilings and a total SU
ceiling. No envelope exists until the user writes one; nothing here creates or implies approval.

The ledger is an append-only JSONL file (path declared in the envelope) mutated only under an
exclusive fcntl lock. Events:
  reserve  launcher reserves SU for one job attempt (time cap x rate) before rendering a script
  start    the runtime marks a reservation used when the job starts; a reservation is single-use,
           so every retry needs a new reservation and counts against the envelope
  settle   consumed SU recorded afterwards (e.g. from sacct); replaces the reserved amount
Committed SU = settled consumption, else the full reservation. Reservations are refused when
committed SU would exceed the phase or total ceiling. Slurm is never submitted from here.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import math
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

from ..constants import ALLOCATION, REMOTE_ROOT
from ..hashing import sha256_file
from .errors import GateError

APPROVAL_SCHEMA = "gn_compute_approval_v1"
LEDGER_SCHEMA = "gn_su_ledger_v1"
SU_EPS = 1e-9  # float-representation slack only (e.g. 1/6 + 5/6); never a budget allowance
PARTITION = "GPU-shared"
GRES = "gpu:h100-80:1"

# v2: the envelope binds to the user's recorded approval receipt instead of pins psc.approved_su_budget.
APPROVAL_SCHEMA_V2 = "gn_compute_approval_v2"
APPROVAL_SCHEMA_V3 = "gn_compute_approval_v3"
USER_APPROVAL_CAP_SU = 60.0                     # provenance/USER_COMPUTE_APPROVAL_20261022.txt
USER_APPROVAL_QUOTE = '"once you are ready to prceed I approve 60 SU"'
USER_APPROVAL_CAP_LINE = "Cumulative ceiling 60 SU"
VERIFIED_SU_PER_GPU_HOUR = 2.0                  # PSC Bridges-2 guide; provenance/PRIMARY_SOURCE_CHECK_20261022_1415.txt
MAX_JOB_MINUTES = 360                           # user approval: maximum six hours per job
MAX_CONCURRENT_GPUS = 1                         # user approval: at most one H100 at a time


def finite_positive(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _strict_load(p: Path) -> Any:
    def no_dup(pairs):
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise GateError([f"GATE approval: {p} has duplicate JSON keys"])
        return dict(pairs)

    def no_const(name):
        raise GateError([f"GATE approval: {p} contains nonfinite constant {name}"])

    return json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=no_dup, parse_constant=no_const)


def _v2_problems(a: Mapping, phases: tuple[str, ...], project_cap_su: Any) -> list[str]:
    probs = []
    src = a.get("source_approval")
    if not isinstance(src, Mapping) or not isinstance(src.get("path"), str) or not isinstance(src.get("sha256"), str):
        probs.append("source_approval {path, sha256} of the user's recorded approval receipt is required")
    else:
        sp = Path(src["path"])
        if not sp.is_file():
            probs.append(f"source approval receipt {sp} does not exist")
        elif sha256_file(sp) != src["sha256"]:
            probs.append(f"source approval receipt {sp} sha256 differs from the envelope")
        else:
            text = sp.read_text(encoding="utf-8")
            if USER_APPROVAL_QUOTE not in text or USER_APPROVAL_CAP_LINE not in text:
                probs.append("source approval receipt does not contain the recorded 60 SU approval")
    if a.get("su_per_gpu_hour") != VERIFIED_SU_PER_GPU_HOUR or isinstance(a.get("su_per_gpu_hour"), bool):
        probs.append(f"su_per_gpu_hour must be the verified rate {VERIFIED_SU_PER_GPU_HOUR}")
    total = a.get("total_su")
    if finite_positive(total):
        if total > USER_APPROVAL_CAP_SU:
            probs.append(f"total_su {total} exceeds the user-approved {USER_APPROVAL_CAP_SU} SU")
        if finite_positive(project_cap_su) and total > project_cap_su:
            probs.append(f"total_su {total} exceeds psc.approved_su_budget {project_cap_su}")
    if a.get("max_job_minutes") != MAX_JOB_MINUTES:
        probs.append(f"max_job_minutes must be {MAX_JOB_MINUTES}")
    if a.get("max_concurrent_gpus") != MAX_CONCURRENT_GPUS:
        probs.append(f"max_concurrent_gpus must be {MAX_CONCURRENT_GPUS}")
    for name, spec in (a.get("phases") or {}).items() if isinstance(a.get("phases"), Mapping) else ():
        if isinstance(spec, Mapping) and finite_positive(spec.get("max_wall_minutes")) \
                and spec["max_wall_minutes"] > MAX_JOB_MINUTES:
            probs.append(f"phase {name}: max_wall_minutes exceeds {MAX_JOB_MINUTES}")
    return probs


def is_v3(approval: Mapping) -> bool:
    return approval.get("schema") == APPROVAL_SCHEMA_V3


def is_v2(approval: Mapping) -> bool:
    return approval.get("schema") == APPROVAL_SCHEMA_V2


def load_approval(path: str | Path, expected_sha256: str, phases: tuple[str, ...],
                  project_cap_su: Any = None) -> dict:
    """Validated envelope; raises GateError naming every problem.

    v1: capped by pins psc.approved_su_budget (required). v2: capped by the user's recorded approval
    receipt (path + sha256 inside the envelope), the verified rate, 360-minute jobs and one GPU at a
    time; a set psc.approved_su_budget is additionally enforced."""
    p = Path(path)
    if not p.exists():
        raise GateError([f"GATE approval: envelope {p} does not exist (no approval has been granted)"])
    if sha256_file(p) != expected_sha256:
        raise GateError([f"GATE approval: envelope {p} sha256 differs from the launch reference"])
    a = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(a, dict):
        raise GateError([f"GATE approval: envelope {p} is not a JSON object"])
    if a.get("schema") == APPROVAL_SCHEMA_V3:
        a = _strict_load(p)
        from ..interventions.aggregate import validate_authority
        from .gates import PHASES as baseline_phases
        from ..interventions.approval import HYBRID_PHASES
        a = _validate_common(a, tuple(dict.fromkeys(baseline_phases + HYBRID_PHASES)), None, validate_authority(a))
        a["_envelope_path"] = str(p.resolve())
        a["_aggregate_phases"] = tuple(dict.fromkeys(baseline_phases + HYBRID_PHASES))
        return a
    if a.get("schema") == APPROVAL_SCHEMA_V2:
        a = _strict_load(p)  # v2: duplicate keys and NaN/Infinity constants are refused at parse time
        return _validate_common(a, phases, None, _v2_problems(a, phases, project_cap_su))
    return _validate_common(a, phases, project_cap_su, [])


def _validate_common(a: dict, phases: tuple[str, ...], project_cap_su: Any, probs: list[str]) -> dict:
    v2 = is_v2(a) or is_v3(a)
    if a.get("schema") not in (APPROVAL_SCHEMA, APPROVAL_SCHEMA_V2, APPROVAL_SCHEMA_V3):
        probs.append(f"schema must be {APPROVAL_SCHEMA} or {APPROVAL_SCHEMA_V2}")
    for f in ("approval_id", "approved_by", "approved_at", "ledger_path"):
        if not isinstance(a.get(f), str) or not a[f].strip():
            probs.append(f"{f} missing")
    for f, want in (("account", ALLOCATION), ("partition", PARTITION), ("gres", GRES), ("remote_root", REMOTE_ROOT)):
        if a.get(f) != want:
            probs.append(f"{f} must be {want!r}")
    for f in ("total_su", "su_per_gpu_hour"):
        if not finite_positive(a.get(f)):
            probs.append(f"{f} must be a finite positive number")
    if finite_positive(a.get("total_su")) and not v2:
        if not finite_positive(project_cap_su):
            probs.append("pins psc.approved_su_budget is unset; the envelope cannot be checked against it")
        elif a["total_su"] > project_cap_su:
            probs.append(f"total_su {a['total_su']} exceeds psc.approved_su_budget {project_cap_su}")
    ph = a.get("phases")
    if not isinstance(ph, Mapping) or not ph:
        probs.append("phases must enumerate the approved phases")
    else:
        for name, spec in ph.items():
            if name not in phases:
                probs.append(f"unknown phase {name!r}")
            elif not isinstance(spec, Mapping) or not finite_positive(spec.get("max_su")) or \
                    not finite_positive(spec.get("max_wall_minutes")):
                probs.append(f"phase {name}: max_su and max_wall_minutes must be finite positive numbers")
            elif finite_positive(a.get("total_su")) and spec["max_su"] > a["total_su"]:
                probs.append(f"phase {name}: max_su exceeds total_su")
    if probs:
        raise GateError([f"GATE approval: {m}" for m in probs])
    return a


def job_su(approval: Mapping, time_limit_minutes: float, gpus: int = 1) -> float:
    return float(time_limit_minutes) / 60.0 * float(approval["su_per_gpu_hour"]) * gpus


@contextlib.contextmanager
def _locked(ledger_path: Path):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(ledger_path) + ".lock", "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def read_events(ledger_path: str | Path) -> list[dict]:
    p = Path(ledger_path)
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise _bad(i, f"unparseable ledger line ({exc})") from exc
    return out


def _append(ledger_path: Path, event: dict) -> None:
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
        fh.flush()


def _bad(i: int, msg: str) -> GateError:
    return GateError([f"GATE ledger: line {i + 1}: {msg}; ledger replay refused"])


def accounting(events: list[dict], approval_id: str, approval_sha256: str, strict: bool = False, parallel: bool = False) -> dict:
    """Replay with strict validation: known schema/events only, one reservation per id, at most one
    start and one settle, finite SU (reserve > 0, consumed >= 0), and every event for this
    approval_id bound to the SAME envelope hash (a changed envelope under one id is refused).

    strict (v2 envelopes): the ledger belongs to exactly one approval, so an event for any other
    approval_id is refused, and every event needs a unique event_id (a replayed/duplicated line is
    refused). Duplicate event_ids are refused whenever present."""
    res: dict[str, dict] = {}
    seen_events: set[str] = set()
    for i, e in enumerate(events):
        if not isinstance(e, dict) or e.get("schema") != LEDGER_SCHEMA or e.get("event") not in ("reserve", "submission", "start", "settle"):
            raise _bad(i, f"unknown schema/event {e.get('event') if isinstance(e, dict) else e!r}")
        eid = e.get("event_id")
        if eid is not None or strict:
            if not isinstance(eid, str) or not eid:
                raise _bad(i, "missing event_id")
            if eid in seen_events:
                raise _bad(i, f"duplicate event_id {eid}")
            seen_events.add(eid)
        if e.get("approval_id") != approval_id:
            if strict:
                raise _bad(i, f"event for unknown approval_id {e.get('approval_id')!r} in a single-approval ledger")
            continue
        rid = e.get("reservation_id")
        if not isinstance(rid, str) or not rid:
            raise _bad(i, "missing reservation_id")
        if e["event"] == "reserve":
            if rid in res:
                raise _bad(i, f"duplicate reservation {rid}")
            if e.get("approval_sha256") != approval_sha256:
                raise _bad(i, "approval envelope changed under the same approval_id")
            if not finite_positive(e.get("su")):
                raise _bad(i, f"reservation {rid} has nonfinite or nonpositive su")
            after = e.get("after_job")
            opened = [r for r in res.values() if r["consumed_su"] is None]
            if after is not None and not valid_job_id(after):
                raise _bad(i, "invalid predecessor allocation identity")
            if after is not None and not parallel:
                tails = [r for r in opened if not any(c.get("after_job") == (r.get("submission") or {}).get("slurm_job_id")
                                                    for c in opened if c is not r)]
                if not valid_job_id(after) or len(tails) != 1 or (tails[0].get("submission") or {}).get("slurm_job_id") != after:
                    raise _bad(i, "predecessor is not the recorded open chain tail")
            if strict and not parallel and opened and after is None:
                raise _bad(i, "open reservations require a serial predecessor")
            res[rid] = {**e, "started": None, "submission": None, "consumed_su": None}
        elif rid not in res:
            raise _bad(i, f"{e['event']} for unknown reservation {rid}")
        elif e["event"] == "submission":
            job = e.get("slurm_job_id")
            if not valid_job_id(job) or res[rid]["submission"] is not None or res[rid]["consumed_su"] is not None:
                raise _bad(i, "invalid or duplicate submission")
            if any((r.get("submission") or {}).get("slurm_job_id") == job for r in res.values()):
                raise _bad(i, "duplicate Slurm job identity")
            started = res[rid]["started"]
            if started is not None and started.get("slurm_job_id") != job:
                raise _bad(i, "submission differs from runtime start")
            res[rid]["submission"] = e
        elif e["event"] == "start":
            if res[rid]["started"] is not None:
                raise _bad(i, f"duplicate start for {rid}")
            res[rid]["started"] = e
        else:
            c = e.get("consumed_su")
            if res[rid]["consumed_su"] is not None:
                raise _bad(i, f"duplicate settle for {rid}")
            if not (isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and c >= 0):
                raise _bad(i, f"settle for {rid} has nonfinite or negative consumed_su")
            res[rid]["consumed_su"] = float(c)
    committed = {rid: (r["consumed_su"] if r["consumed_su"] is not None else float(r["su"])) for rid, r in res.items()}
    by_phase: dict[str, float] = {}
    for rid, r in res.items():
        by_phase[r["phase"]] = by_phase.get(r["phase"], 0.0) + committed[rid]
    return {"reservations": res, "committed_total": sum(committed.values()), "committed_by_phase": by_phase,
            "open_reservations": sorted(rid for rid, r in res.items() if r["consumed_su"] is None),
            "over_reservation": sorted(rid for rid, r in res.items()
                                       if r["consumed_su"] is not None and r["consumed_su"] > float(r["su"]) + SU_EPS)}


def ledger_accounting(approval: Mapping, approval_sha256: str) -> dict:
    return accounting(read_events(approval["ledger_path"]), approval["approval_id"], approval_sha256,
                      strict=is_v2(approval) or is_v3(approval), parallel=is_v3(approval))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _event(kind: str, approval: Mapping, reservation_id: str, **fields) -> dict:
    return {"schema": LEDGER_SCHEMA, "event": kind, "event_id": uuid.uuid4().hex, "reservation_id": reservation_id,
            "approval_id": approval["approval_id"], **fields, "at": _now()}


def valid_job_id(job: Any) -> bool:
    return isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job) is not None


def record_submission(approval: Mapping, approval_sha256: str, reservation_id: str, job_id: str) -> dict:
    if not valid_job_id(job_id):
        raise GateError(["GATE ledger: job-id must be a positive numeric Slurm allocation ID"])
    ledger = Path(approval["ledger_path"])
    with _locked(ledger):
        acct = ledger_accounting(approval, approval_sha256)
        r = acct["reservations"].get(reservation_id)
        if r is None or r["consumed_su"] is not None:
            raise GateError(["GATE ledger: submission requires an open reservation"])
        event = _event("submission", approval, reservation_id, slurm_job_id=job_id)
        accounting(read_events(ledger) + [event], approval["approval_id"], approval_sha256, strict=is_v2(approval) or is_v3(approval), parallel=is_v3(approval))
        _append(ledger, event)
    return event


def predecessor_terminal(job_id: str) -> bool:
    """Positive scheduler evidence only; absence/errors remain closed. No SU is released here."""
    if not valid_job_id(job_id):
        return False
    try:
        result = subprocess.run(["sacct", "-n", "-P", "-X", "-j", job_id, "--format=JobIDRaw,State"],
                                capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    matches = [r for r in rows if len(r) >= 2 and r[0] == job_id]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    return len(matches) == 1 and matches[0][1].split()[0] in terminal


def reserve(approval: Mapping, approval_sha256: str, phase: str, time_limit_minutes: float, note: str = "", after_job: str | None = None) -> dict:
    if is_v3(approval):
        from ..interventions import aggregate
        return aggregate.reserve(approval, approval_sha256, phase, time_limit_minutes, note=note, after_job=after_job)
    if phase not in approval["phases"]:
        raise GateError([f"GATE approval: phase {phase!r} is not enumerated in the approval envelope"])
    spec = approval["phases"][phase]
    if not finite_positive(time_limit_minutes) or time_limit_minutes > spec["max_wall_minutes"]:
        raise GateError([f"GATE approval: time limit {time_limit_minutes} min exceeds phase max_wall_minutes "
                         f"{spec['max_wall_minutes']}"])
    if (is_v2(approval) or is_v3(approval)) and time_limit_minutes > MAX_JOB_MINUTES:
        raise GateError([f"GATE approval: time limit {time_limit_minutes} min exceeds {MAX_JOB_MINUTES} min per job"])
    su = job_su(approval, time_limit_minutes)
    ledger = Path(approval["ledger_path"])
    with _locked(ledger):
        acct = accounting(read_events(ledger), approval["approval_id"], approval_sha256, strict=is_v2(approval) or is_v3(approval), parallel=is_v3(approval))
        if after_job is not None:
            if not valid_job_id(after_job):
                raise GateError(["GATE queue: invalid predecessor job ID"])
            opened = [r for r in acct["reservations"].values() if r["consumed_su"] is None]
            tails = [r for r in opened if not any(c.get("after_job") == (r.get("submission") or {}).get("slurm_job_id")
                                                for c in opened if c is not r)]
            if not is_v3(approval) and (len(tails) != 1 or (tails[0].get("submission") or {}).get("slurm_job_id") != after_job):
                raise GateError(["GATE queue: predecessor must be the recorded open chain tail"])
        if is_v2(approval) and acct["open_reservations"] and after_job is None:
            raise GateError([f"GATE budget: one GPU job at a time; reconcile/settle open reservation(s) "
                             f"{acct['open_reservations']} before reserving another"])
        if acct["committed_total"] + su > approval["total_su"] + SU_EPS:
            raise GateError([f"GATE budget: reserving {su:.2f} SU exceeds envelope total "
                             f"({acct['committed_total']:.2f} of {approval['total_su']} committed)"])
        if acct["committed_by_phase"].get(phase, 0.0) + su > spec["max_su"] + SU_EPS:
            raise GateError([f"GATE budget: reserving {su:.2f} SU exceeds phase {phase} max_su {spec['max_su']}"])
        event = _event("reserve", approval, uuid.uuid4().hex, approval_sha256=approval_sha256, phase=phase,
                       time_limit_minutes=time_limit_minutes, su=su, note=note, after_job=after_job)
        _append(ledger, event)
    return event


def check_reservation(approval: Mapping, approval_sha256: str, reservation_id: Any, phase: str,
                      time_limit_minutes: float, after_job: str | None = None) -> list[str]:
    """Read-only gate check: reservation exists, unused, unsettled, matches phase and covers the cap."""
    if not isinstance(reservation_id, str) or not reservation_id:
        return ["GATE reservation: authorization.reservation_id missing (launcher must reserve SU first)"]
    try:
        acct = ledger_accounting(approval, approval_sha256)
    except GateError as exc:
        return exc.problems
    r = acct["reservations"].get(reservation_id)
    if r is None:
        return [f"GATE reservation: {reservation_id} not found in the ledger"]
    probs = []
    if r.get("after_job") != after_job:
        probs.append("GATE reservation: predecessor differs from this launch")
    if r["approval_sha256"] != approval_sha256 or r["phase"] != phase:
        probs.append("GATE reservation: approval/phase differ from this launch")
    if r["started"] is not None or r["consumed_su"] is not None:
        probs.append("GATE reservation: already used; every attempt (including retries) needs a new reservation")
    if not finite_positive(time_limit_minutes) or job_su(approval, time_limit_minutes) > float(r["su"]) + 1e-9:
        probs.append("GATE reservation: reserved SU does not cover this job's time cap")
    return probs


def mark_started(approval: Mapping, approval_sha256: str, reservation_id: str, phase: str,
                 time_limit_minutes: float, job_id: str) -> dict:
    if is_v3(approval):
        from ..interventions import aggregate
        return aggregate.mark_started(approval, approval_sha256, reservation_id, phase, time_limit_minutes, job_id)
    ledger = Path(approval["ledger_path"])
    with _locked(ledger):
        acct = ledger_accounting(approval, approval_sha256)
        r = acct["reservations"].get(reservation_id, {})
        probs = check_reservation(approval, approval_sha256, reservation_id, phase, time_limit_minutes, r.get("after_job"))
        if (r.get("after_job") or r.get("submission")) and (r.get("submission") or {}).get("slurm_job_id") != job_id:
            probs.append("GATE queue: runtime job must match recorded submission")
        if r.get("after_job") and not predecessor_terminal(r["after_job"]):
            probs.append("GATE queue: predecessor is not positively terminal")
        running_others = 0
        for other_id, other in acct["reservations"].items():
            if other_id != reservation_id and other["started"] is not None and other["consumed_su"] is None:
                prior_job = (other.get("submission") or {}).get("slurm_job_id")
                if not prior_job or not predecessor_terminal(prior_job):
                    running_others += 1
        if running_others >= (approval["max_concurrent_gpus"] if is_v3(approval) else 1):
            probs.append("GATE queue: maximum concurrent GPU count would be exceeded")
        if probs:
            raise GateError(probs)
        event = _event("start", approval, reservation_id, slurm_job_id=job_id)
        _append(ledger, event)
    return event


def settle(approval: Mapping, approval_sha256: str, reservation_id: str, consumed_su: float, source: str,
           **extra) -> dict:
    if not (isinstance(consumed_su, (int, float)) and not isinstance(consumed_su, bool)
            and math.isfinite(consumed_su) and consumed_su >= 0):
        raise GateError(["GATE ledger: consumed_su must be finite and nonnegative"])
    if not isinstance(source, str) or not source.strip():
        raise GateError(["GATE ledger: settle source (accounting evidence) is required"])
    ledger = Path(approval["ledger_path"])
    with _locked(ledger):
        r = ledger_accounting(approval, approval_sha256)["reservations"].get(reservation_id)
        if r is None or r["consumed_su"] is not None:
            raise GateError([f"GATE ledger: reservation {reservation_id} unknown or already settled"])
        started_job = (r["started"] or r.get("submission") or {}).get("slurm_job_id")
        if extra.get("slurm_job_id") is not None and started_job is not None and extra["slurm_job_id"] != started_job:
            raise GateError([f"GATE ledger: scheduler job {extra['slurm_job_id']} differs from the job "
                             f"{started_job} that started reservation {reservation_id}"])
        event = _event("settle", approval, reservation_id, consumed_su=float(consumed_su), source=source, **extra)
        _append(ledger, event)
    return event


def reconcile(approval: Mapping, approval_sha256: str, reservation_id: str, slurm_job_id: str,
              elapsed_seconds: Any, alloc_gpus: Any, source: str) -> dict:
    """Settle from scheduler accounting (e.g. sacct ElapsedRaw and the allocated GPU count):
    consumed = elapsed_hours x verified rate x GPUs. Unknown or malformed accounting is refused, never
    guessed; a job that never ran is settled with elapsed 0 and its scheduler evidence as source."""
    if not isinstance(slurm_job_id, str) or not slurm_job_id.strip():
        raise GateError(["GATE ledger: slurm_job_id (scheduler accounting identity) is required"])
    if type(elapsed_seconds) is not int or elapsed_seconds < 0:
        raise GateError(["GATE ledger: elapsed_seconds must be a nonnegative integer (sacct ElapsedRaw)"])
    if type(alloc_gpus) is not int or alloc_gpus not in (0, MAX_CONCURRENT_GPUS):
        raise GateError([f"GATE ledger: alloc_gpus must be 0 or {MAX_CONCURRENT_GPUS} (one H100 per job)"])
    consumed = elapsed_seconds / 3600.0 * float(approval["su_per_gpu_hour"]) * alloc_gpus
    return settle(approval, approval_sha256, reservation_id, consumed,
                  f"scheduler accounting: {source}", slurm_job_id=slurm_job_id, elapsed_seconds=elapsed_seconds,
                  alloc_gpus=alloc_gpus)


def instantiate_approval(template_path: str | Path, out_path: str | Path, phases: tuple[str, ...]) -> tuple[dict, str]:
    """Write the immutable v2 envelope from its authored template by recording the SHA256 of the
    user's approval receipt named in the template. Write-once: an existing envelope is never
    replaced (no envelope-hash reset under the same approval_id)."""
    from ..io_utils import write_once_json

    t = json.loads(Path(template_path).read_text(encoding="utf-8"))
    if t.get("schema") != APPROVAL_SCHEMA_V2:
        raise GateError([f"GATE approval: template schema must be {APPROVAL_SCHEMA_V2}"])
    src = t.get("source_approval") or {}
    sp = Path(src.get("path") or "")
    if not sp.is_file():
        raise GateError([f"GATE approval: source approval receipt {sp} does not exist"])
    if src.get("sha256") not in (None, sha256_file(sp)):
        raise GateError(["GATE approval: template records a different source approval sha256"])
    t = {k: v for k, v in t.items() if not k.startswith("_")}
    t["source_approval"] = {"path": str(src["path"]), "sha256": sha256_file(sp)}
    sha = write_once_json(out_path, t)
    return load_approval(out_path, sha, phases), sha
