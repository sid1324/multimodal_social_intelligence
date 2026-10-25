"""B1 clean SFT core (plan 7, A3), independent of Hugging Face so CPU tests run the same code.

Objective per accumulation window of W NormLens records (each paired with one NormBank draw):
    L = (1/W) sum_i CE_N(i) + 0.25 (1/W) sum_i CE_T(i)
with CE the full-vocabulary FP32 vote CE at the last prompt position (objective.vote_ce); no EOS
or rationale label exists anywhere. Graphs are backpropagated sequentially per record (NormLens
then NormBank), each pre-divided by the ACTUAL window size W (so the merged final window of
16+r is normalized by 16+r); then one grad-clip(1.0) / AdamW / scheduler step.

Run control (TrainingRun): epoch-boundary checkpoints only feed selection; each epoch checkpoint
records its dev evaluation as pending, and the completed dev result is persisted write-once. The
time cap is checked before EVERY update (reserving measured update+checkpoint time) and before
every dev evaluation (reserving measured dev time), independent of epoch boundaries. A pending
dev evaluation is always completed on resume before any further update or selection.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..constants import (ADAM_BETAS, ADAM_EPS, GRAD_CLIP, LR, NORMBANK_LOSS_WEIGHT, WARMUP_FRACTION, WEIGHT_DECAY)
from ..objective import vote_ce
from .errors import CheckpointError, NonfiniteError, RuntimeContractError

LR_SCHEDULE_VERSION = "warmup_ceil5pct_linear_from_1_over_W__cosine_from_1_no_terminal_zero_v1"

# forward(task, input_record) -> (logits_last [V] tensor, answer_token_ids)
ForwardFn = Callable[[str, Mapping], tuple]


def warmup_steps(total_updates: int) -> int:
    return max(1, math.ceil(WARMUP_FRACTION * total_updates))


def lr_multiplier(update: int, total_updates: int, warmup: int) -> float:
    """Multiplier for 0-based optimizer update `update` (implementation-defined, versioned).

    Warmup: (u+1)/W for u < W (no wasted zero-LR first update; update W-1 reaches 1).
    Cosine: 0.5(1+cos(pi (u-W)/(T-W))) for W <= u < T (first post-warmup update at 1, last > 0).
    u == T is the scheduler's terminal state after the final scheduler.step(); no optimizer
    update ever consumes it, so it is defined as 0.0 rather than raising."""
    if update == total_updates:
        return 0.0
    if not 0 <= update < total_updates:
        raise ValueError("update index outside the locked horizon")
    if update < warmup:
        return (update + 1) / warmup
    span = total_updates - warmup
    return 0.5 * (1.0 + math.cos(math.pi * (update - warmup) / span)) if span > 0 else 1.0


def build_optimizer(params: Sequence):
    import torch

    return torch.optim.AdamW(list(params), lr=LR, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
                             foreach=False)


def build_scheduler(optimizer, total_updates: int):
    import torch

    w = warmup_steps(total_updates)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: lr_multiplier(u, total_updates, w))


def group_updates(schedule: Mapping) -> list[list[dict]]:
    """Rows grouped by optimizer update; validates contiguity and window sizes."""
    if schedule.get("arm") != "B1":
        raise RuntimeContractError("baseline trainer accepts only the B1 schedule")
    updates: list[list[dict]] = []
    for row in schedule["rows"]:
        if row.get("site_layer") is not None or row.get("base_entry") is not None:
            raise RuntimeContractError("B1 schedule must not contain intervention sites")
        if row["update"] == len(updates):
            updates.append([])
        elif row["update"] != len(updates) - 1:
            raise RuntimeContractError("schedule updates are not contiguous")
        updates[-1].append(row)
    for u, rows in enumerate(updates):
        if any(r["window_size"] != len(rows) for r in rows) or \
                [r["position_in_window"] for r in rows] != list(range(len(rows))):
            raise RuntimeContractError(f"update {u}: window rows inconsistent with window_size")
        if len({r["epoch"] for r in rows}) != 1:
            raise RuntimeContractError(f"update {u} spans epochs")
    if len(updates) != schedule["total_updates"]:
        raise RuntimeContractError("schedule total_updates mismatch")
    return updates


def _finite(t, what: str) -> None:
    import torch

    if not bool(torch.isfinite(t).all()):
        raise NonfiniteError(f"nonfinite {what}; hard stop")


class B1Trainer:
    def __init__(self, named_params: Sequence[tuple[str, object]], forward: ForwardFn, schedule: Mapping,
                 normlens_inputs: Mapping[str, Mapping], normlens_targets: Mapping[str, Sequence[float]],
                 normbank_inputs: Mapping[str, Mapping], normbank_targets: Mapping[str, Sequence[float]]):
        self.named_params = sorted(named_params, key=lambda x: x[0])
        if not self.named_params or not all(p.requires_grad for _, p in self.named_params):
            raise RuntimeContractError("trainer needs a nonempty set of trainable parameters")
        self.params = [p for _, p in self.named_params]
        self.forward = forward
        self.schedule = schedule
        self.updates = group_updates(schedule)
        self.total_updates = len(self.updates)
        self.epochs = int(schedule["epochs"])
        self.nl_in, self.nl_q = normlens_inputs, normlens_targets
        self.nb_in, self.nb_q = normbank_inputs, normbank_targets
        for r in schedule["rows"]:
            if r["normlens_id"] not in self.nl_in or r["normbank_id"] not in self.nb_in:
                raise RuntimeContractError(f"schedule row {r['ordinal']} references an unknown example")
        self.optimizer = build_optimizer(self.params)
        self.scheduler = build_scheduler(self.optimizer, self.total_updates)

    @property
    def param_names(self) -> list[str]:
        return [n for n, _ in self.named_params]

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def epoch_of(self, u: int) -> int:
        return int(self.updates[u][0]["epoch"])

    def window_loss_terms(self, rows: Sequence[Mapping]) -> tuple[list[float], list[float]]:
        """Sequential weighted backward over one window; returns per-record CE values."""
        w = len(rows)
        ce_n, ce_t = [], []
        for r in rows:
            logits, ids = self.forward("normlens", self.nl_in[r["normlens_id"]])
            _finite(logits, f"NormLens logits ({r['normlens_id']})")  # before the shared objective's own guard
            loss_n = vote_ce(logits, ids, self.nl_q[r["normlens_id"]])
            _finite(loss_n, f"NormLens CE ({r['normlens_id']})")
            (loss_n / w).backward()
            ce_n.append(float(loss_n.detach()))
            logits_t, ids_t = self.forward("normbank", self.nb_in[r["normbank_id"]])
            _finite(logits_t, f"NormBank logits ({r['normbank_id']})")
            loss_t = vote_ce(logits_t, ids_t, self.nb_q[r["normbank_id"]])
            _finite(loss_t, f"NormBank CE ({r['normbank_id']})")
            (NORMBANK_LOSS_WEIGHT * loss_t / w).backward()
            ce_t.append(float(loss_t.detach()))
        return ce_n, ce_t

    def run_update(self, u: int) -> dict:
        import torch

        if self.scheduler.last_epoch != u or not 0 <= u < self.total_updates:
            raise RuntimeContractError(f"scheduler at {self.scheduler.last_epoch}, asked to run update {u}")
        rows = self.updates[u]
        t0 = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        ce_n, ce_t = self.window_loss_terms(rows)
        for n, p in self.named_params:
            if p.grad is None:
                raise RuntimeContractError(f"trainable parameter {n} received no gradient")
            _finite(p.grad, f"gradient of {n}")
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP, error_if_nonfinite=True)
        lr = self.current_lr()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        mn, mt = sum(ce_n) / len(ce_n), sum(ce_t) / len(ce_t)
        return {"update": u, "epoch": rows[0]["epoch"], "window_size": len(rows), "lr": lr,
                "mean_ce_normlens": mn, "mean_ce_normbank": mt, "loss": mn + NORMBANK_LOSS_WEIGHT * mt,
                "grad_norm_preclip": float(gnorm), "seconds": time.monotonic() - t0}

    def is_epoch_end(self, u: int) -> bool:
        return u + 1 == self.total_updates or self.epoch_of(u + 1) != self.epoch_of(u)


def select_epoch(dev_results: Sequence[Mapping], epochs: int) -> dict:
    """Epoch-boundary dev conditional NLL; minimum wins, exact ties prefer the earlier epoch."""
    seen = [int(r["epoch"]) for r in dev_results]
    if len(seen) != len(set(seen)):
        raise RuntimeContractError(f"duplicate dev results for epochs {sorted(seen)}")
    by_epoch = {int(r["epoch"]): float(r["cond_nll"]) for r in dev_results}
    if sorted(by_epoch) != list(range(epochs)):
        raise RuntimeContractError(f"dev results for epochs {sorted(by_epoch)}, expected 0..{epochs - 1}")
    if not all(math.isfinite(v) for v in by_epoch.values()):
        raise NonfiniteError("nonfinite dev NLL; selection stopped")
    best = 0
    for e in range(1, epochs):
        if by_epoch[e] < by_epoch[best]:
            best = e
    return {"selected_epoch": best, "dev_cond_nll": {str(k): v for k, v in by_epoch.items()},
            "rule": "min dev cond NLL, ties -> earlier epoch", "choice_exists": epochs > 1}


class Budget:
    """Wall-clock cap for this process (job time limit minus stop margin)."""

    def __init__(self, limit_seconds: float, clock: Callable[[], float] = time.monotonic):
        if not (isinstance(limit_seconds, (int, float)) and math.isfinite(limit_seconds) and limit_seconds > 0):
            raise RuntimeContractError("time budget must be a finite positive number of seconds")
        self.limit, self.clock, self.start = float(limit_seconds), clock, clock()

    def remaining(self) -> float:
        return self.limit - (self.clock() - self.start)

    def allows(self, reserve_seconds: float) -> bool:
        return self.remaining() >= reserve_seconds


# dev_fn(epoch, checkpoint_info) -> {"cond_nll": float, ...} or {"status": "paused", ...};
# the model is at that checkpoint's state. A pause must leave the epoch pending.
DevFn = Callable[[int, Mapping], Mapping]


class TrainingRun:
    """Checkpointed B1 run with persisted dev results and exact pause/resume semantics."""

    def __init__(self, run_dir: str | Path, identity: Mapping, trainer: B1Trainer, dev_fn: DevFn,
                 budget: Budget | None = None, update_reserve_s: float = 0.0, dev_reserve_s: float = 0.0):
        for v in (update_reserve_s, dev_reserve_s):
            if not math.isfinite(v) or v < 0:
                raise RuntimeContractError("reserves must be finite and nonnegative")
        self.run_dir = Path(run_dir)
        self.identity = dict(identity)
        self.trainer = trainer
        self.dev_fn = dev_fn
        self.budget = budget
        self.update_reserve = float(update_reserve_s)
        self.dev_reserve = float(dev_reserve_s)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.dev_root = self.run_dir / "dev_results"
        self.next_update = 0
        self.log: list = []
        self.pending: tuple[int, dict] | None = None
        self.latest_ckpt: dict | None = None
        self._ckpt_seconds = 0.0

    # -------------------------------------------------------------- open / resume
    def open(self, resume: bool) -> None:
        from ..io_utils import read_json, write_once_json
        from .checkpoint import load_checkpoint, resume_trainer

        if not resume:
            if self.run_dir.exists():
                raise CheckpointError(f"{self.run_dir} exists; use resume")
            self.run_dir.mkdir(parents=True)
            write_once_json(self.run_dir / "identity.json", self.identity)
            return
        if read_json(self.run_dir / "identity.json", verify=True) != self.identity:
            raise CheckpointError("resume identity differs from the original run")
        cks = sorted(p for p in self.ckpt_root.iterdir() if p.is_dir()) if self.ckpt_root.exists() else []
        if not cks:
            # Paused before the first update: nothing was trained, so restart from the identical
            # (identity-verified, seeded) initial state. Any dev result here would be inconsistent.
            if self.dev_root.exists() and any(self.dev_root.iterdir()):
                raise CheckpointError("dev results exist without any checkpoint")
            return
        ck = load_checkpoint(cks[-1], trusted_root=self.run_dir)
        self.next_update = resume_trainer(self.trainer, ck, self.identity)
        self.log = list(ck["state"]["update_log"])
        self.latest_ckpt = ck["info"]
        st = ck["state"]
        pending = st["extra"].get("dev_pending_epoch")
        if pending is not None and not self.dev_done(pending):
            self.pending = (int(pending), ck["info"])
        completed = st["epoch"] + 1 if st["kind"] == "epoch" else st["epoch"]
        for e in range(completed):
            if not self.dev_done(e) and (self.pending is None or self.pending[0] != e):
                raise CheckpointError(f"epoch {e} finished but its dev evaluation is neither done nor pending")

    def dev_done(self, epoch: int) -> bool:
        from ..io_utils import read_json

        path = self.dev_root / f"epoch{epoch}.json"
        # A crash after the metric file but before its sidecar must re-enter deterministic
        # dev finalization to verify identical bytes and complete the sidecar.
        if not path.exists() or not path.with_name(path.name + ".sha256").exists():
            return False
        record = read_json(path, verify=True)
        if record.get("epoch") != epoch:
            raise CheckpointError(f"dev record for epoch {epoch} belongs to another epoch")
        return True

    # -------------------------------------------------------------- pieces
    def _save(self, next_update: int, epoch: int, kind: str) -> dict:
        from .checkpoint import save_checkpoint

        t0 = time.monotonic()
        info = save_checkpoint(self.ckpt_root / f"u{next_update:05d}_{kind}", trainer=self.trainer,
                               next_update=next_update, epoch=epoch, kind=kind, identity=self.identity, log=self.log,
                               extra={"dev_pending_epoch": epoch if kind == "epoch" else None})
        self._ckpt_seconds = max(self._ckpt_seconds, time.monotonic() - t0)
        self.latest_ckpt = info
        return info

    def _run_dev(self, epoch: int, ck: Mapping) -> dict | None:
        from .evaluate import _json_bytes, _write_or_verify

        t0 = time.monotonic()
        res = dict(self.dev_fn(epoch, ck))
        if res.get("status") == "paused":
            return None
        nll = res.get("cond_nll")
        if not isinstance(nll, (int, float)) or not math.isfinite(nll):
            raise NonfiniteError(f"dev evaluation for epoch {epoch} returned no finite cond_nll")
        record = {"epoch": epoch, "cond_nll": float(nll), "checkpoint": dict(ck),
                  "details": {k: v for k, v in res.items() if k != "cond_nll"}}
        json.dumps(record)
        _write_or_verify(self.dev_root / f"epoch{epoch}.json", _json_bytes(record))
        self.dev_reserve = max(self.dev_reserve, time.monotonic() - t0)
        return record

    def _paused(self, reason: str) -> dict:
        return {"status": "paused", "reason": reason, "next_update": self.next_update,
                "pending_dev_epoch": self.pending[0] if self.pending else None,
                "checkpoint": self.latest_ckpt}

    # -------------------------------------------------------------- main loop
    def run(self, max_updates: int | None = None) -> dict:
        t = self.trainer
        if self.pending is not None:
            if self.budget is not None and not self.budget.allows(self.dev_reserve):
                return self._paused("time cap before pending dev evaluation")
            if self._run_dev(*self.pending) is None:
                return self._paused("time cap during pending dev evaluation")
            self.pending = None
        done, unsaved = 0, False
        while self.next_update < t.total_updates:
            u = self.next_update
            reserve = self.update_reserve + self._ckpt_seconds
            capped = self.budget is not None and not self.budget.allows(reserve)
            if capped or (max_updates is not None and done >= max_updates):
                if unsaved:
                    self._save(u, t.epoch_of(u - 1), "pause")
                return self._paused("time cap" if capped else "max_updates")
            stats = t.run_update(u)
            self.log.append(stats)
            self.update_reserve = max(self.update_reserve, stats["seconds"])
            self.next_update, done, unsaved = u + 1, done + 1, True
            if t.is_epoch_end(u):
                ck = self._save(u + 1, t.epoch_of(u), "epoch")
                unsaved = False
                self.pending = (t.epoch_of(u), ck)
                if self.budget is not None and not self.budget.allows(self.dev_reserve):
                    return self._paused("time cap before dev evaluation")
                if self._run_dev(*self.pending) is None:
                    return self._paused("time cap during dev evaluation")
                self.pending = None
        return {"status": "complete", "next_update": self.next_update, "checkpoint": self.latest_ckpt}

    def select(self) -> dict:
        from ..io_utils import read_json, write_once_json

        if self.next_update != self.trainer.total_updates or self.pending is not None:
            raise RuntimeContractError("selection requires the completed horizon and no pending dev evaluation")
        results = [read_json(p, verify=True) for p in sorted(self.dev_root.glob("epoch*.json"))]
        sel = select_epoch(results, self.trainer.epochs)
        chosen = next(r for r in results if int(r["epoch"]) == sel["selected_epoch"])
        sel["checkpoint"] = chosen["checkpoint"]
        write_once_json(self.run_dir / "selection.json", {**sel, "identity": self.identity})
        return sel
