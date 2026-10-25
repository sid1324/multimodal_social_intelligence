"""Local-only Qwen3.5-9B model/processor loading and the native forward (plan 6.2, 8.1, 8.2).

* Offline: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE set before import, local_files_only=True, a pinned
  snapshot directory, safetensors only, trust_remote_code=False. Source/config/files are
  verified (pins.py) BEFORE from_pretrained is called.
* Backends: explicit eager attention for the text full-attention layers and the vision tower;
  GatedDeltaNet uses the pinned source's pure-PyTorch chunk path (fla/causal_conv1d/kernels must
  be absent so no hub/fused kernel can be dispatched). The same policy serves B0, B1 training,
  dev selection, evaluation and export.
* Native forward: batch 1, no padding, use_cache=False, model rope_deltas reset per example, the
  ViT/merger run in eval mode under torch.no_grad() (their features enter through the native
  ``mm_encoder_outputs`` argument, consumed as ``pooler_output`` by the pinned source), the
  decoder keeps autograd, and only the final prompt row's logits are materialized
  (``logits_to_keep=1``). No hooks are installed anywhere.
* Special tokens: ``<|im_end|>`` (assistant-turn terminator, the tokenizer's eos_token) and the
  text config's ``eos_token_id`` are distinct native tokens; both are recorded, neither is a
  training target, and the terminator is retained only for Ego candidate scoring elsewhere.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from typing import Mapping

from ..constants import MAX_PIXELS, MAX_PROMPT_TOKENS, MIN_PIXELS, MODEL_CLASS, VISUAL_TOKENS_MAX, VISUAL_TOKENS_MIN
from .errors import InventoryError, PinError, RuntimeContractError
from .pins import (NATIVE_FALLBACK_FORBIDDEN_MODULES, assert_model_config, receipt_chat_template, receipt_config,
                   receipt_tokenizer_config, verify_imported_modeling_module, verify_model_dir, verify_processor_dir,
                   verify_runtime_packages, verify_transformers)

ATTN_IMPLEMENTATION = "eager"
MIN_GPU_BYTES = 79 * 10 ** 9
TURN_TERMINATOR = "<|im_end|>"
HASH_CHUNK_ELEMENTS = 1 << 24
_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
TASK_LETTER_COUNT = {"normlens": 3, "normlens_shuffle": 3, "normlens_text_only": 3, "normbank": 3}
IMAGE_TASKS = ("normlens", "normlens_shuffle")


def prepare_process_env() -> dict:
    """Call before importing torch/transformers. Refuses conflicting online settings."""
    for k, v in _OFFLINE_ENV.items():
        if os.environ.get(k, v) != v:
            raise PinError(f"environment {k}={os.environ[k]!r} conflicts with required {v!r}")
        os.environ[k] = v
    return dict(_OFFLINE_ENV)


def set_determinism() -> dict:
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": True, "cudnn_deterministic": True, "tf32": False,
            "note": "warn_only: ops without deterministic kernels warn; resume is exact only where kernels are"}


def check_device() -> dict:
    """Exactly one visible H100 80 GB before any weights are touched."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeContractError("GATE device: CUDA unavailable; GPU phases run only on the authorized H100")
    if torch.cuda.device_count() != 1:
        raise RuntimeContractError(f"GATE device: {torch.cuda.device_count()} GPUs visible, expected exactly 1")
    props = torch.cuda.get_device_properties(0)
    if "H100" not in props.name or props.total_memory < MIN_GPU_BYTES:
        raise RuntimeContractError(f"GATE device: {props.name} with {props.total_memory} bytes is not an H100 80 GB")
    return {"name": props.name, "total_memory": props.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def delta_dispatch_inventory() -> dict:
    """Pinned native fallback only: no fused/hub kernel module may be importable."""
    present = {m: importlib.util.find_spec(m) is not None for m in NATIVE_FALLBACK_FORBIDDEN_MODULES}
    if any(present.values()):
        raise PinError(f"kernel packages present {present}: the native-fallback Delta policy forbids them "
                       "(a different backend needs its own pin and reviewed equivalence tests)")
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    funcs = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update", "eager_attention_forward"):
        target = getattr(getattr(m, name), "__wrapped__", getattr(m, name))
        funcs[name] = {"module": getattr(target, "__module__", None), "qualname": getattr(target, "__qualname__", None)}
    return {"policy": "torch_native_fallback", "kernel_modules_present": present, "functions": funcs}


# ------------------------------------------------------------------ processor
def configure_image_budget(processor, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> dict:
    """A3 pixel budget via image_processor.size={shortest_edge, longest_edge} (the pinned API;
    min_pixels/max_pixels are deprecated and not used). Effective behavior is verified by
    processing images (processor_check)."""
    ip = processor.image_processor

    def size_dict():
        s = getattr(ip, "size", None)
        if isinstance(s, Mapping):
            return dict(s)
        return {k: v for k, v in vars(s).items() if v is not None} if hasattr(s, "__dict__") else {}

    before = size_dict()
    if "shortest_edge" not in before or "longest_edge" not in before:
        raise PinError(f"image processor size {before} lacks shortest_edge/longest_edge")
    ip.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    after = size_dict()
    if after.get("shortest_edge") != min_pixels or after.get("longest_edge") != max_pixels:
        raise PinError(f"image budget did not take effect: {after}")
    return {"before": before, "after": after}


def check_processor_tokens(processor, model_cfg: Mapping, tok_receipt: Mapping) -> dict:
    """Config-defined vision ids must match; tokenizer eos/pad strings must match the pinned
    tokenizer_config; the turn terminator and the config eos id are recorded as distinct tokens."""
    tok = processor.tokenizer
    want = {processor.image_token: model_cfg["image_token_id"], "<|vision_start|>": model_cfg["vision_start_token_id"],
            "<|vision_end|>": model_cfg["vision_end_token_id"]}
    got = {k: tok.convert_tokens_to_ids(k) for k in want}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        raise PinError(f"processor vision token ids differ from the pinned config (got, want): {bad}")
    for attr in ("eos_token", "pad_token"):
        if getattr(tok, attr) != tok_receipt[attr]:
            raise PinError(f"tokenizer {attr} {getattr(tok, attr)!r} != pinned tokenizer_config {tok_receipt[attr]!r}")
    if TURN_TERMINATOR not in receipt_chat_template_text(processor):
        raise PinError(f"chat template does not use {TURN_TERMINATOR} as the turn terminator")
    ip = processor.image_processor
    if int(ip.merge_size) != 2 or int(getattr(ip, "patch_size", 16)) != 16:
        raise PinError("processor merge_size/patch_size differ from the pinned vision config")
    cfg_eos = model_cfg["text_config"]["eos_token_id"]
    return {
        "vision_token_ids": got,
        "assistant_turn_terminator": {"token": TURN_TERMINATOR, "id": tok.convert_tokens_to_ids(TURN_TERMINATOR),
                                      "use": "Ego candidate scoring only; never a B0/B1 training target"},
        "tokenizer_eos": {"token": tok.eos_token, "id": tok.eos_token_id},
        "tokenizer_pad": {"token": tok.pad_token, "id": tok.pad_token_id},
        "config_text_eos": {"id": cfg_eos, "token": tok.convert_ids_to_tokens(cfg_eos)},
        "generation_config_in_revision": "generation_config.json" in tok_receipt["siblings"],
        "merge_size": int(ip.merge_size), "patch_size": int(getattr(ip, "patch_size", 16)),
    }


def receipt_chat_template_text(processor) -> str:
    return getattr(processor, "chat_template", None) or getattr(processor.tokenizer, "chat_template", None) or ""


def excluded_answer_ids(token_report: Mapping) -> set[int]:
    """Special tokens that may never serve as answer tokens (no EOS/terminator targets)."""
    ids = {token_report["assistant_turn_terminator"]["id"], token_report["config_text_eos"]["id"],
           token_report["tokenizer_eos"]["id"], token_report["tokenizer_pad"]["id"]}
    return {int(i) for i in ids if i is not None}


def load_processor(processor_dir: str, pins: Mapping, require_pin: bool):
    assets = verify_processor_dir(pins, processor_dir, require_pin=require_pin)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True, trust_remote_code=False)
    if receipt_chat_template_text(processor) != receipt_chat_template(pins):
        raise PinError("processor chat template differs from the pinned revision's template")
    report = {"assets": assets, "image_budget": configure_image_budget(processor),
              "tokens": check_processor_tokens(processor, receipt_config(pins), receipt_tokenizer_config(pins)),
              "chat_template_matches_receipt": True}
    return processor, report


def validate_serialized(ser, task: str, excluded_ids: set[int], expected_answer_ids: list[int] | None = None) -> None:
    """Runtime re-validation of Serializer output before any forward."""
    n = len(ser.input_ids)
    problems = []
    if n > MAX_PROMPT_TOKENS or ser.final_prompt_row != n - 1:
        problems.append(f"{n} tokens / final row {ser.final_prompt_row}")
    ids = [int(i) for i in ser.answer_token_ids]
    if len(ids) != TASK_LETTER_COUNT[task] or len(set(ids)) != len(ids):
        problems.append(f"answer ids {ids}")
    if set(ids) & excluded_ids:
        problems.append(f"answer ids {ids} include special/EOS/terminator ids")
    if expected_answer_ids is not None and ids != list(expected_answer_ids):
        problems.append(f"answer ids {ids} differ from earlier examples {expected_answer_ids}")
    if task in IMAGE_TASKS:
        if not VISUAL_TOKENS_MIN <= ser.num_visual_tokens <= VISUAL_TOKENS_MAX or \
                len(ser.image_positions) != ser.num_visual_tokens:
            problems.append(f"visual tokens {ser.num_visual_tokens}/{len(ser.image_positions)}")
    elif ser.image_positions or ser.num_visual_tokens:
        problems.append("image tokens in a text-only prompt")
    if task.startswith("normlens"):
        r = ser.completed_action_row
        if r is None or not r < ser.final_prompt_row or ser.decoded_action_span is None or \
                (ser.image_positions and r <= ser.image_positions[-1]):
            problems.append(f"completed-action row {r}")
    if problems:
        raise RuntimeContractError(f"{ser.example_id} ({task}): serializer metadata invalid: {problems}")


# ------------------------------------------------------------------ model
def base_of(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def assert_native_model(model, check_config: bool = True) -> dict:
    """Class, config, eager backends at text FA and vision, and per-layer mixer families."""
    base = base_of(model)
    if type(base).__name__ != MODEL_CLASS:
        raise InventoryError(f"loaded {type(base).__name__}, expected {MODEL_CLASS}")
    if check_config:
        assert_model_config(base.config)
    impls = {"text": base.config.text_config._attn_implementation,
             "vision": base.config.vision_config._attn_implementation}
    if any(v != ATTN_IMPLEMENTATION for v in impls.values()):
        raise InventoryError(f"attention implementations {impls}; explicit eager required for text FA and vision")
    for name, mod in base.named_modules():
        cfg = getattr(mod, "config", None)
        if type(mod).__name__ in ("Qwen3_5Attention", "Qwen3_5VisionAttention") and \
                getattr(cfg, "_attn_implementation", None) != ATTN_IMPLEMENTATION:
            raise InventoryError(f"{name} does not dispatch eager attention")
    families = []
    types = list(base.config.text_config.layer_types)
    for i, layer in enumerate(base.model.language_model.layers):
        if types[i] == "full_attention":
            ok = type(getattr(layer, "self_attn", None)).__name__ == "Qwen3_5Attention" and not hasattr(layer, "linear_attn")
        else:
            ok = type(getattr(layer, "linear_attn", None)).__name__ == "Qwen3_5GatedDeltaNet" and \
                type(layer.linear_attn.norm).__name__ == "Qwen3_5RMSNormGated" and not hasattr(layer, "self_attn")
        if not ok:
            raise InventoryError(f"layer {i} mixer class differs from the pinned native implementation")
        families.append("F" if types[i] == "full_attention" else "D")
    return {"attn_implementation": impls, "families": "".join(families)}


def load_model(model_dir: str, pins: Mapping, device: str = "cuda"):
    """Verify pins/files first, then load the BF16 base locally with eager attention, frozen."""
    tf = verify_transformers(pins)
    files = verify_model_dir(pins, model_dir)
    pkgs = verify_runtime_packages(pins)
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    modeling_sha = verify_imported_modeling_module(pins)
    dispatch = delta_dispatch_inventory()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
        use_safetensors=True, trust_remote_code=False)
    native = assert_native_model(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    model.eval()
    base_of(model).model.rope_deltas = None
    return model, {"transformers": tf, "model_files": files, "packages": pkgs, "modeling_sha256": modeling_sha,
                   "delta_dispatch": dispatch, "native": native}


def set_train_mode(model) -> None:
    """Decoder in train mode (dropout is zero everywhere); ViT/merger eval and frozen."""
    model.train()
    visual = base_of(model).model.visual
    visual.eval()
    if visual.training or any(p.requires_grad for p in visual.parameters()):
        raise RuntimeContractError("vision tower must be eval/frozen during SFT")


def frozen_state_sha256(model, chunk_elements: int = HASH_CHUNK_ELEMENTS) -> dict:
    """Exact byte-level SHA256 over every frozen tensor (name, dtype, shape, raw bytes), streamed
    in chunks so no large host or FP64 copy is made. Meta tensors are refused."""
    import torch

    h = hashlib.sha256()
    n = 0
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda x: x[0]):
            if p.requires_grad:
                continue
            if p.is_meta:
                raise RuntimeContractError(f"{name} is a meta tensor; frozen-state hashing needs materialized weights")
            h.update(f"{name}|{p.dtype}|{tuple(p.shape)}|".encode())
            flat = p.detach().reshape(-1)
            for start in range(0, flat.numel(), chunk_elements):
                h.update(flat[start:start + chunk_elements].contiguous().view(torch.uint8).cpu().numpy().tobytes())
            n += 1
    return {"sha256": h.hexdigest(), "frozen_tensors": n}


class NativeForward:
    """Batch-1 native forward returning the native-dtype logits row at the final prompt position."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def __call__(self, ser, grad: bool):
        import torch

        n = len(ser.input_ids)
        if ser.final_prompt_row != n - 1:
            raise RuntimeContractError("final prompt row must be the last input position (batch-1, no padding)")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in dict(ser.model_inputs).items()}
        if tuple(inputs["input_ids"].shape) != (1, n) or inputs["input_ids"][0].tolist() != list(ser.input_ids):
            raise RuntimeContractError("serialized input_ids differ from the model inputs")
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.bool().all()):
            raise RuntimeContractError("padding present in a batch-1 prompt")
        base = base_of(self.model)
        base.model.rope_deltas = None  # no state reused across examples
        mm = None
        pixel_values = inputs.pop("pixel_values", None)
        if pixel_values is not None:
            with torch.no_grad():
                mm = {"image": base.model.get_image_features(pixel_values, inputs["image_grid_thw"], return_dict=True)}
        with (torch.enable_grad() if grad else torch.no_grad()):
            out = self.model(**inputs, mm_encoder_outputs=mm, use_cache=False, logits_to_keep=1)
        logits = out.logits
        if tuple(logits.shape[:2]) != (1, 1):
            raise RuntimeContractError(f"unexpected logits shape {tuple(logits.shape)}")
        return logits[0, -1]


class TaskForward:
    """Serializer + images + NativeForward for the trainer/evaluator (input-only records).
    Every serialized prompt is re-validated; answer IDs must be identical across examples."""

    def __init__(self, serializer, native: NativeForward, images, grad: bool, excluded_ids: set[int],
                 shuffle_map: Mapping | None = None):
        self.serializer, self.native, self.images, self.grad = serializer, native, images, grad
        self.excluded_ids = set(excluded_ids)
        self.shuffle_map = shuffle_map
        self.answer_ids: dict[str, list[int]] = {}

    def serialize(self, task: str, rec: Mapping):
        if task in IMAGE_TASKS:
            src = rec if task == "normlens" else {"example_id": rec["example_id"], **self.shuffle_map[rec["example_id"]]}
            ser = self.serializer.normlens(rec["example_id"], rec["action"], self.images.load(src), text_only=False)
        elif task == "normlens_text_only":
            ser = self.serializer.normlens(rec["example_id"], rec["action"], None, text_only=True)
        elif task == "normbank":
            ser = self.serializer.normbank(rec["example_id"], rec["setting"], rec["behavior"], rec["constraints"])
        else:
            raise RuntimeContractError(f"unknown task {task!r}")
        kind = "normbank" if task == "normbank" else "normlens"
        validate_serialized(ser, task, self.excluded_ids, self.answer_ids.get(kind))
        self.answer_ids.setdefault(kind, [int(i) for i in ser.answer_token_ids])
        return ser

    def __call__(self, task: str, rec: Mapping):
        ser = self.serialize(task, rec)
        return self.native(ser, grad=self.grad), ser.answer_token_ids

    def with_meta(self, task: str):
        def fn(rec: Mapping):
            ser = self.serialize(task, rec)
            return self.native(ser, grad=False), ser.answer_token_ids, ser.metadata()
        return fn
