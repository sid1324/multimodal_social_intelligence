"""Immutable runtime pins (plan 6.2, A6), read from configs/runtime_pins.json (manager schema).

* Transformers: the Git commit is read from the installed distribution's PEP 610
  direct_url.json (or an editable checkout's .git files) and the installed modeling_qwen3_5.py
  must hash to the pinned SHA256; the "5.21.0.dev0" version string is necessary, not sufficient.
* Model: local snapshot only, tied to the pinned HF commit via blob ids / local-dir metadata,
  config.json byte-hash pinned, safetensors-only, per-shard SHA256 pinned.
* Null fields in the pins file are OPEN GATES; any GPU/model phase refuses while one is open.
Nothing here downloads anything or imports transformers.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from ..constants import (DELTA_CONV_WIDTH, DELTA_HEAD_DIM, DELTA_NUM_K_HEADS, DELTA_NUM_V_HEADS, FA_HEAD_DIM,
                         FA_NUM_KV_HEADS, FA_NUM_Q_HEADS, FULL_ATTENTION_LAYERS, HIDDEN_SIZE, MODEL_CLASS, NUM_LAYERS)
from ..hashing import canonical_json_bytes, git_blob_sha1, sha256_bytes, sha256_file
from .errors import PinError

TRANSFORMERS_VERSION = "5.21.0.dev0"
SUPPORTED_DELTA_BACKENDS = ("torch_native_fallback",)
NATIVE_FALLBACK_FORBIDDEN_MODULES = ("fla", "causal_conv1d", "kernels")
PACKAGE_PINS = ("torch", "peft", "accelerate", "safetensors")
FORBIDDEN_MODEL_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".py", ".pyc", ".msgpack", ".h5",
                            ".npy", ".npz", ".joblib")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5")
MODELING_RELPATH = Path("models") / "qwen3_5" / "modeling_qwen3_5.py"
CHAT_TEMPLATE_RECEIPT = "provenance/native_source/hf-model-metadata.json"
# Processor/tokenizer/template files in the pinned revision's sibling listing (hf-model-metadata.json).
PROCESSOR_FILES = ("chat_template.jinja", "merges.txt", "preprocessor_config.json", "tokenizer.json",
                   "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json")


def project_root() -> Path:
    # src/grounded_norm_sft/runtime/pins.py -> project root
    return Path(__file__).resolve().parents[3]


def default_pins_path() -> Path:
    return project_root() / "configs" / "runtime_pins.json"


def load_pins(path: str | Path | None = None) -> tuple[dict, str]:
    path = Path(path) if path else default_pins_path()
    if not path.exists():
        raise PinError(f"runtime pins file missing: {path}")
    pins = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "transformers", "runtime_gates_open", "psc"):
        if not isinstance(pins.get(key), dict):
            raise PinError(f"runtime pins lack section {key!r}")
    for sec, k in (("transformers", "git_sha"), ("transformers", "modeling_qwen3_5_sha256"),
                   ("model", "hf_commit_sha"), ("model", "config_sha256"), ("model", "config_source")):
        if not pins[sec].get(k):
            raise PinError(f"runtime pins lack {sec}.{k}")
    return pins, sha256_file(path)


def open_pin_gates(pins: Mapping) -> list[str]:
    """Null pins that block any model/GPU execution."""
    out = [f"runtime_gates_open.{k}" for k, v in pins["runtime_gates_open"].items() if v in (None, "")]
    for k in ("processor_and_chat_template_sha256", "safetensors_shard_sha256"):
        if pins["model"].get(k) in (None, "", {}):
            out.append(f"model.{k}")
    if pins["psc"].get("approved_su_budget") in (None, "", 0):
        out.append("psc.approved_su_budget")
    return out


# ------------------------------------------------------------------ transformers
def _git_head(repo_dir: Path) -> str | None:
    """Read HEAD commit from .git files (no shell)."""
    git = repo_dir / ".git"
    if git.is_file():  # worktree pointer "gitdir: ..."
        txt = git.read_text().strip()
        if txt.startswith("gitdir:"):
            git = (repo_dir / txt.split(":", 1)[1].strip()).resolve()
    head = git / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if not ref.startswith("ref:"):
        return ref
    name = ref.split(":", 1)[1].strip()
    if (git / name).exists():
        return (git / name).read_text().strip()
    packed = git / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == name:
                return parts[0]
    return None


def transformers_install_info() -> dict:
    """Version, recorded VCS commit and installed modeling file path, without importing transformers."""
    try:
        dist = importlib.metadata.distribution("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PinError("transformers is not installed in this environment") from exc
    direct = dist.read_text("direct_url.json")
    direct_url = json.loads(direct) if direct else None
    commit, source = None, None
    if direct_url and isinstance(direct_url.get("vcs_info"), dict):
        commit, source = direct_url["vcs_info"].get("commit_id"), "direct_url.vcs_info"
    elif direct_url and direct_url.get("url", "").startswith("file://"):
        commit, source = _git_head(Path(direct_url["url"][len("file://"):])), "editable_checkout_.git"
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.origin:
        raise PinError("cannot locate the installed transformers package")
    return {"version": dist.version, "commit": commit, "commit_source": source, "direct_url": direct_url,
            "modeling_path": str(Path(spec.origin).parent / MODELING_RELPATH)}


def verify_transformers(pins: Mapping, info: Mapping | None = None) -> dict:
    info = dict(info) if info is not None else transformers_install_info()
    tp = pins["transformers"]
    problems = []
    if info.get("version") != TRANSFORMERS_VERSION:
        problems.append(f"transformers version {info.get('version')!r} != {TRANSFORMERS_VERSION!r}")
    if not info.get("commit"):
        problems.append("installed transformers records no Git commit (install from the pinned commit so PEP 610 "
                        "direct_url.json has vcs_info.commit_id); the version string alone is insufficient")
    elif info["commit"] != tp["git_sha"]:
        problems.append(f"transformers commit {info['commit']} != pinned {tp['git_sha']}")
    modeling = Path(info["modeling_path"])
    actual = sha256_file(modeling) if modeling.exists() else None
    if actual != tp["modeling_qwen3_5_sha256"]:
        problems.append(f"installed modeling_qwen3_5.py sha256 {actual} != pinned {tp['modeling_qwen3_5_sha256']}")
    if problems:
        raise PinError("PIN transformers: " + "; ".join(problems))
    return {**info, "modeling_sha256": actual}


def verify_imported_modeling_module(pins: Mapping) -> str:
    """After import: the module object actually in use must be the pinned file."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    actual = sha256_file(m.__file__)
    if actual != pins["transformers"]["modeling_qwen3_5_sha256"]:
        raise PinError(f"PIN transformers: imported {m.__file__} sha256 {actual} is not the pinned source")
    return actual


def verify_runtime_packages(pins: Mapping) -> dict:
    """Installed torch/peft/accelerate/safetensors/CUDA must equal the (non-null) pins."""
    gates = pins["runtime_gates_open"]
    problems, report = [], {}
    for name in PACKAGE_PINS:
        want = gates.get(name)
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have = None
        report[name] = have
        if want in (None, ""):
            problems.append(f"runtime_gates_open.{name} is null (installed: {have})")
        elif str(want).split("==")[-1] != have:
            problems.append(f"{name} installed {have} != pinned {want}")
    import torch

    report["cuda"] = torch.version.cuda
    if gates.get("cuda") in (None, ""):
        problems.append(f"runtime_gates_open.cuda is null (torch built with CUDA {torch.version.cuda})")
    elif str(gates["cuda"]) != str(torch.version.cuda):
        problems.append(f"torch CUDA {torch.version.cuda} != pinned {gates['cuda']}")
    backend = gates.get("delta_kernel_backend")
    if backend not in SUPPORTED_DELTA_BACKENDS:
        problems.append(f"runtime_gates_open.delta_kernel_backend={backend!r}; this slice supports only "
                        f"{SUPPORTED_DELTA_BACKENDS} (other backends need reviewed dispatch/equivalence tests)")
    for k in ("gpu_driver", "container_or_env_hash"):
        if gates.get(k) in (None, ""):
            problems.append(f"runtime_gates_open.{k} is null")
        report[k + "_attested"] = gates.get(k)
    if problems:
        raise PinError("PIN runtime packages: " + "; ".join(problems))
    return report


# ------------------------------------------------------------------ model config
def _get(cfg: Any, key: str):
    return cfg.get(key) if isinstance(cfg, Mapping) else getattr(cfg, key, None)


def assert_model_config(cfg: Any) -> dict:
    """Architecture facts the plan relies on (sections 3, 6.1). Accepts a dict or a loaded config."""
    text, vision = _get(cfg, "text_config"), _get(cfg, "vision_config")
    if text is None or vision is None:
        raise PinError("model config lacks text_config/vision_config")
    expected = {
        "num_hidden_layers": NUM_LAYERS, "hidden_size": HIDDEN_SIZE, "num_attention_heads": FA_NUM_Q_HEADS,
        "num_key_value_heads": FA_NUM_KV_HEADS, "head_dim": FA_HEAD_DIM, "linear_num_key_heads": DELTA_NUM_K_HEADS,
        "linear_num_value_heads": DELTA_NUM_V_HEADS, "linear_key_head_dim": DELTA_HEAD_DIM,
        "linear_value_head_dim": DELTA_HEAD_DIM, "linear_conv_kernel_dim": DELTA_CONV_WIDTH,
        "attention_dropout": 0.0, "attn_output_gate": True,
    }
    problems = [f"text_config.{k}={_get(text, k)!r} != {v!r}" for k, v in expected.items() if _get(text, k) != v]
    layer_types = list(_get(text, "layer_types") or [])
    fa = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    if fa != FULL_ATTENTION_LAYERS or len(layer_types) != NUM_LAYERS or \
            any(t not in ("full_attention", "linear_attention") for t in layer_types):
        problems.append(f"layer_types full-attention indices {fa} != {FULL_ATTENTION_LAYERS}")
    arch = _get(cfg, "architectures")
    if arch is not None and list(arch) != [MODEL_CLASS]:
        problems.append(f"architectures {arch} != [{MODEL_CLASS}]")
    if list(_get(vision, "deepstack_visual_indexes") or []) != []:
        problems.append("vision deepstack_visual_indexes must be empty")
    if _get(vision, "spatial_merge_size") != 2 or _get(vision, "patch_size") != 16:
        problems.append("vision patch_size/spatial_merge_size differ from inspected 16/2")
    if problems:
        raise PinError("model config assertion failed: " + "; ".join(problems))
    return {"full_attention_layers": list(fa), "num_layers": len(layer_types)}


def receipt_config(pins: Mapping) -> dict:
    return json.loads((project_root() / pins["model"]["config_source"]).read_text(encoding="utf-8"))


def receipt_metadata(pins: Mapping) -> dict:
    meta = json.loads((project_root() / CHAT_TEMPLATE_RECEIPT).read_text(encoding="utf-8"))
    if meta.get("sha") != pins["model"]["hf_commit_sha"]:
        raise PinError("model metadata receipt is not from the pinned model revision")
    return meta


def receipt_chat_template(pins: Mapping) -> str:
    return receipt_metadata(pins)["config"]["tokenizer_config"]["chat_template"]


def receipt_tokenizer_config(pins: Mapping) -> dict:
    """Pinned tokenizer_config identities (eos/pad token strings) and the revision file listing."""
    meta = receipt_metadata(pins)
    tc = meta["config"]["tokenizer_config"]
    return {"eos_token": tc.get("eos_token"), "pad_token": tc.get("pad_token"), "bos_token": tc.get("bos_token"),
            "siblings": sorted(s["rfilename"] for s in meta.get("siblings", []))}


# ------------------------------------------------------------------ files / revision integrity
def _blob_ok(path: Path, blob_id: str) -> bool:
    if len(blob_id) == 64:
        return sha256_file(path) == blob_id
    if len(blob_id) == 40:
        return git_blob_sha1(path.read_bytes()) == blob_id
    return False


def hf_file_identity(root: Path, rel: Path, revision: str) -> dict:
    """Tie one file to the pinned revision.

    HF cache layout: snapshots/<revision>/<file> is a symlink to blobs/<sha256|git-sha1>.
    --local-dir layout: .cache/huggingface/download/<file>.metadata holds commit then etag."""
    f = root / rel
    if root.name == revision and root.parent.name == "snapshots":
        if not f.is_symlink():
            raise PinError(f"{rel}: snapshot file is not a blob symlink; cannot tie it to the revision")
        blob = f.resolve()
        if not _blob_ok(blob, blob.name):
            raise PinError(f"{rel}: content does not match its blob id {blob.name}")
        return {"blob": blob.name, "layout": "hf_cache"}
    meta = root / ".cache" / "huggingface" / "download" / (str(rel) + ".metadata")
    if meta.exists():
        lines = meta.read_text().splitlines()
        if len(lines) < 2 or lines[0].strip() != revision:
            raise PinError(f"{rel}: local-dir metadata commit is not the pinned revision")
        etag = lines[1].strip().strip('"')
        if not _blob_ok(f, etag):
            raise PinError(f"{rel}: content does not match recorded etag {etag}")
        return {"blob": etag, "layout": "hf_local_dir"}
    raise PinError(f"{rel}: no HF revision evidence (need snapshots/<rev> symlinks or .cache/huggingface metadata)")


def _listing(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*")
                  if (p.is_file() or p.is_symlink()) and ".cache" not in p.relative_to(root).parts)


def verify_model_dir(pins: Mapping, model_dir: str | Path) -> dict:
    """Local-only, revision-tied, safetensors-only snapshot; config/shard hashes equal the pins."""
    root = Path(model_dir)
    if not root.is_dir():
        raise PinError(f"model_dir does not exist: {root} (weights are never downloaded by this runtime)")
    rev = pins["model"]["hf_commit_sha"]
    files = _listing(root)
    bad = [str(f) for f in files if f.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES]
    if bad:
        raise PinError(f"model_dir contains pickle/code files that are never loaded: {bad}")
    for req in ("config.json", "model.safetensors.index.json"):
        if Path(req) not in files:
            raise PinError(f"model_dir lacks {req}")
    if sha256_file(root / "config.json") != pins["model"]["config_sha256"]:
        raise PinError("model_dir/config.json sha256 differs from model.config_sha256")
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if cfg != receipt_config(pins):
        raise PinError("model_dir/config.json differs from the provenance config receipt")
    assert_model_config(cfg)
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing = [s for s in shards if Path(s) not in files]
    if not shards or missing:
        raise PinError(f"safetensors shards missing: {missing or 'empty weight_map'}")
    shard_pins = pins["model"].get("safetensors_shard_sha256")
    if not isinstance(shard_pins, Mapping) or sorted(shard_pins) != shards:
        raise PinError("model.safetensors_shard_sha256 must map exactly the index shards to SHA256 (open gate)")
    report = {}
    for rel in files:
        ident = hf_file_identity(root, rel, rev)
        if str(rel) in shard_pins:
            digest = sha256_file(root / rel)
            if digest != shard_pins[str(rel)]:
                raise PinError(f"{rel}: sha256 {digest} != pinned {shard_pins[str(rel)]}")
            ident["sha256"] = digest
        report[str(rel)] = ident
    return {"model_dir": str(root), "revision": rev, "files": report, "shards": shards}


def processor_assets_digest(processor_dir: str | Path) -> tuple[str, dict]:
    """Definition of model.processor_and_chat_template_sha256 used by this runtime: SHA256 of
    canonical JSON {file name: file SHA256} over PROCESSOR_FILES (the processor/tokenizer/template
    files of the pinned revision listing), so a processor-only directory and the model snapshot
    hash identically. Every listed file must exist."""
    root = Path(processor_dir)
    missing = [n for n in PROCESSOR_FILES if not (root / n).exists()]
    if missing:
        raise PinError(f"processor assets missing: {missing}")
    files = {n: sha256_file(root / n) for n in PROCESSOR_FILES}
    return sha256_bytes(canonical_json_bytes(files)), files


def verify_processor_dir(pins: Mapping, processor_dir: str | Path, require_pin: bool) -> dict:
    """Processor-only assets (no weights/code). GPU launches require the digest pin to match."""
    root = Path(processor_dir)
    if not root.is_dir():
        raise PinError(f"processor_dir does not exist: {root}")
    code = [str(f) for f in _listing(root) if f.suffix.lower() in (".py", ".pyc", ".pkl", ".pickle")]
    if code:
        raise PinError(f"processor_dir contains code/pickle files: {code}")
    try:
        digest, files = processor_assets_digest(root)
    except PinError as exc:
        if require_pin:
            raise
        return {"processor_dir": str(root), "assets_digest": None, "digest_error": str(exc), "matches_pin": False}
    pinned = pins["model"].get("processor_and_chat_template_sha256")
    if require_pin and digest != pinned:
        raise PinError(f"processor assets digest {digest} != pinned model.processor_and_chat_template_sha256 {pinned}")
    return {"processor_dir": str(root), "assets_digest": digest, "pinned_digest": pinned,
            "matches_pin": digest == pinned, "files": files}
