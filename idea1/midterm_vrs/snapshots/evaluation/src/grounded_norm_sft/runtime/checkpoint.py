"""Complete resume checkpoints without pickle (plan 7, 9.B.9).

A checkpoint directory is write-once and contains:
  adapter.safetensors    trainable LoRA tensors in their FP32 training dtype (never downcast)
  optimizer.safetensors  AdamW per-parameter state (step, exp_avg, exp_avg_sq), keyed by name
  rng.safetensors        torch CPU/CUDA generator states (+ numpy MT19937 key)
  state.json             cursor (next update, epoch), scheduler state, optimizer param-group
                         hyper-parameters, python/numpy RNG scalars, run identity hashes, update log
  MANIFEST.json          SHA256 of every file above (each also has a .sha256 sidecar)
Loading requires an explicit trusted root, verifies every hash, refuses unknown/pickle files and
checks the run identity (schedule, launch-independent config, pins, adapter init) before use.
Bitwise weight identity after resume is promised only where kernels are deterministic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping

from ..hashing import sha256_file
from ..io_utils import read_json, write_once_json
from .errors import CheckpointError
from .tensorio import load_tensors, save_tensors

CKPT_SCHEMA = "gn_b1_checkpoint_v1"
CKPT_FILES = ("adapter.safetensors", "optimizer.safetensors", "rng.safetensors", "state.json")
_PICKLE_SUFFIXES = (".pt", ".pth", ".bin", ".pkl", ".pickle", ".ckpt")


# ------------------------------------------------------------------ RNG
def capture_rng() -> tuple[dict, dict]:
    import numpy as np
    import torch

    tensors = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        for i, s in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"torch_cuda_{i}"] = s
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    tensors["numpy_keys"] = torch.from_numpy(np.asarray(keys, dtype=np.int64))
    py = random.getstate()
    scalars = {"numpy": {"kind": kind, "pos": int(pos), "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
               "python": [py[0], list(py[1]), py[2]], "cuda_devices": torch.cuda.device_count()
               if torch.cuda.is_available() else 0}
    return tensors, scalars


def restore_rng(tensors: Mapping, scalars: Mapping) -> None:
    import numpy as np
    import torch

    torch.set_rng_state(tensors["torch_cpu"])
    n = scalars.get("cuda_devices", 0)
    if n:
        if not torch.cuda.is_available() or torch.cuda.device_count() != n:
            raise CheckpointError("checkpoint CUDA RNG device count differs from this host")
        torch.cuda.set_rng_state_all([tensors[f"torch_cuda_{i}"] for i in range(n)])
    s = scalars["numpy"]
    np.random.set_state((s["kind"], tensors["numpy_keys"].numpy().astype(np.uint32), s["pos"], s["has_gauss"],
                         s["cached_gaussian"]))
    v, state, gauss = scalars["python"]
    random.setstate((v, tuple(state), gauss))


# ------------------------------------------------------------------ optimizer / scheduler
def optimizer_to_tensors(optimizer, param_names: list[str]) -> tuple[dict, list]:
    sd = optimizer.state_dict()
    if len(sd["param_groups"]) != 1 or sd["param_groups"][0]["params"] != list(range(len(param_names))):
        raise CheckpointError("expected one AdamW param group over the ordered trainable parameters")
    tensors = {}
    for idx, st in sd["state"].items():
        name = param_names[idx]
        for k, v in st.items():
            tensors[f"{name}::{k}"] = v.detach().cpu().clone()
    groups = []
    for g in sd["param_groups"]:
        groups.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()})
    return tensors, groups


def optimizer_from_tensors(optimizer, tensors: Mapping, groups: list, param_names: list[str]) -> None:
    state: dict = {}
    for key, t in tensors.items():
        name, field = key.split("::", 1)
        if name not in param_names:
            raise CheckpointError(f"optimizer state for unknown parameter {name}")
        state.setdefault(param_names.index(name), {})[field] = t
    restored = []
    for g in groups:
        g = dict(g)
        if "betas" in g:
            g["betas"] = tuple(g["betas"])
        restored.append(g)
    optimizer.load_state_dict({"state": state, "param_groups": restored})


def scheduler_state(scheduler) -> dict:
    sd = scheduler.state_dict()
    out = {k: v for k, v in sd.items() if k != "lr_lambdas"}
    json.dumps(out)  # must be JSON-serializable
    return out


def load_scheduler_state(scheduler, state: Mapping) -> None:
    sd = dict(state)
    sd["lr_lambdas"] = [None] * len(scheduler.lr_lambdas)
    scheduler.load_state_dict(sd)


# ------------------------------------------------------------------ save / load
def save_checkpoint(ckpt_dir: str | Path, *, trainer, next_update: int, epoch: int, kind: str, identity: Mapping,
                    log: list, extra: Mapping | None = None) -> dict:
    from .lora import adapter_sha256

    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.exists():
        raise CheckpointError(f"refusing to overwrite checkpoint {ckpt_dir}")
    if not 1 <= int(next_update) <= trainer.total_updates or trainer.scheduler.last_epoch != int(next_update):
        raise CheckpointError("checkpoints are written only after >=1 completed update, at the trainer's cursor "
                              "(a zero-update pause restarts from the identical initial state instead)")
    ckpt_dir.mkdir(parents=True)
    names = trainer.param_names
    adapter = {n: p.detach().cpu().clone() for n, p in trainer.named_params}
    bad = [n for n, t in adapter.items() if str(t.dtype) != "torch.float32"]
    if bad:
        raise CheckpointError(f"adapter tensors not FP32: {bad[:3]}")
    opt_t, groups = optimizer_to_tensors(trainer.optimizer, names)
    rng_t, rng_s = capture_rng()
    hashes = {
        "adapter.safetensors": save_tensors(ckpt_dir / "adapter.safetensors", adapter),
        "optimizer.safetensors": save_tensors(ckpt_dir / "optimizer.safetensors", opt_t),
        "rng.safetensors": save_tensors(ckpt_dir / "rng.safetensors", rng_t),
    }
    state = {"schema": CKPT_SCHEMA, "kind": kind, "next_update": int(next_update), "epoch": int(epoch),
             "total_updates": trainer.total_updates, "param_names": names, "optimizer_param_groups": groups,
             "scheduler": scheduler_state(trainer.scheduler), "rng": rng_s, "identity": dict(identity),
             "adapter_sha256": adapter_sha256(adapter), "update_log": list(log),
             "extra": dict(extra or {})}  # nested: cannot override cursor/identity/hash fields
    hashes["state.json"] = write_once_json(ckpt_dir / "state.json", state)
    write_once_json(ckpt_dir / "MANIFEST.json", {"schema": CKPT_SCHEMA, "files": hashes})
    return _info(ckpt_dir, state)


def _info(ckpt_dir: Path, state: Mapping) -> dict:
    return {"dir": str(ckpt_dir), "kind": state["kind"], "next_update": int(state["next_update"]),
            "epoch": int(state["epoch"]), "manifest_sha256": sha256_file(ckpt_dir / "MANIFEST.json")}


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def verify_checkpoint_dir(ckpt_dir: str | Path, trusted_root: str | Path) -> dict:
    ckpt_dir, root = Path(ckpt_dir).resolve(), Path(trusted_root).resolve()
    if not _within(ckpt_dir, root):
        raise CheckpointError(f"checkpoint {ckpt_dir} is outside the explicit trusted root {root}")
    if not ckpt_dir.is_dir():
        raise CheckpointError(f"checkpoint {ckpt_dir} does not exist")
    present = {p.name for p in ckpt_dir.iterdir()}
    pickles = [n for n in present if n.endswith(_PICKLE_SUFFIXES)]
    if pickles:
        raise CheckpointError(f"checkpoint contains pickle-format files {pickles}; refusing")
    allowed = set(CKPT_FILES) | {"MANIFEST.json"}
    allowed |= {n + ".sha256" for n in allowed}
    if present - allowed:
        raise CheckpointError(f"unexpected files in checkpoint: {sorted(present - allowed)}")
    manifest = read_json(ckpt_dir / "MANIFEST.json", verify=True)
    if manifest.get("schema") != CKPT_SCHEMA or sorted(manifest.get("files", {})) != sorted(CKPT_FILES):
        raise CheckpointError("checkpoint manifest schema/files mismatch")
    for name, digest in manifest["files"].items():
        if sha256_file(ckpt_dir / name) != digest:
            raise CheckpointError(f"checkpoint file {name} hash mismatch (tampered or partial)")
    return manifest


def load_checkpoint(ckpt_dir: str | Path, trusted_root: str | Path) -> dict:
    verify_checkpoint_dir(ckpt_dir, trusted_root)
    ckpt_dir = Path(ckpt_dir)
    state = read_json(ckpt_dir / "state.json", verify=True)
    adapter, _ = load_tensors(ckpt_dir / "adapter.safetensors")
    optim, _ = load_tensors(ckpt_dir / "optimizer.safetensors")
    rng, _ = load_tensors(ckpt_dir / "rng.safetensors")
    return {"state": state, "adapter": adapter, "optimizer": optim, "rng": rng, "info": _info(ckpt_dir, state)}


def resume_trainer(trainer, ckpt: Mapping, identity: Mapping) -> int:
    """Restore adapters/optimizer/scheduler/RNG into a freshly built trainer; return next update."""
    from .lora import adapter_sha256, load_adapter_tensors

    st = ckpt["state"]
    if st["identity"] != dict(identity):
        diff = sorted(k for k in set(st["identity"]) | set(identity) if st["identity"].get(k) != identity.get(k))
        raise CheckpointError(f"checkpoint belongs to a different run (differs in {diff})")
    if st["param_names"] != trainer.param_names or st["total_updates"] != trainer.total_updates:
        raise CheckpointError("checkpoint parameter list or horizon differs from this trainer")
    if adapter_sha256(ckpt["adapter"]) != st["adapter_sha256"]:
        raise CheckpointError("adapter tensor hash mismatch")
    load_adapter_tensors(_ParamView(trainer.named_params), ckpt["adapter"])
    optimizer_from_tensors(trainer.optimizer, ckpt["optimizer"], st["optimizer_param_groups"], trainer.param_names)
    load_scheduler_state(trainer.scheduler, st["scheduler"])
    if trainer.scheduler.last_epoch != st["next_update"]:
        raise CheckpointError("scheduler position differs from the checkpoint cursor")
    restore_rng(ckpt["rng"], st["rng"])
    return int(st["next_update"])


class _ParamView:
    """Adapter so lora.load_adapter_tensors can target the trainer's named parameter list."""

    def __init__(self, named):
        self._named = list(named)

    def named_parameters(self):
        return iter(self._named)
