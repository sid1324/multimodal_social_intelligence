"""All-32-layer native-mixer LoRA (plan 6.1): rank 16, alpha 32, dropout 0, bias none on

  full attention (8 layers): self_attn.{q_proj,k_proj,v_proj,o_proj}
  GatedDeltaNet (24 layers): linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}

Targets are enumerated on the actual module tree before wrapping and must equal the expected
name set exactly; fused Delta projections stop the run. Production expectation is derived from
the pinned config and must equal the locked 18,112,512 (never weakened); tests pass explicitly
scaled tiny-model expectations through the same functions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping, Sequence

from ..constants import (DELTA_LORA_MODULES, DELTA_MIXER_ATTR, EXPECTED_LORA_PARAMS_PER_DELTA_LAYER,
                         EXPECTED_LORA_PARAMS_PER_FA_LAYER, EXPECTED_LORA_PARAMS_TOTAL, FA_LORA_MODULES, FA_MIXER_ATTR,
                         FULL_ATTENTION_LAYERS, FUSED_DELTA_NAMES, LORA_ALPHA, LORA_DROPOUT, LORA_RANK, NUM_LAYERS)
from .errors import InventoryError

TEXT_LAYERS_PREFIX = "model.language_model.layers."
_LORA_PARAM_RE = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+)\.lora_(?P<ab>[AB])\.(?P<adapter>[^.]+)\.weight$")


def target_regex(prefix: str = TEXT_LAYERS_PREFIX) -> str:
    """Full-name regex (PEFT applies re.fullmatch when target_modules is a str)."""
    return (re.escape(prefix) + r"\d+\.(?:" + re.escape(FA_MIXER_ATTR) + r"\.(?:" + "|".join(FA_LORA_MODULES) + r")|"
            + re.escape(DELTA_MIXER_ATTR) + r"\.(?:" + "|".join(DELTA_LORA_MODULES) + r"))")


def expected_target_names(num_layers: int = NUM_LAYERS, fa_layers: Sequence[int] = FULL_ATTENTION_LAYERS,
                          prefix: str = TEXT_LAYERS_PREFIX) -> list[str]:
    out = []
    for i in range(num_layers):
        if i in fa_layers:
            out += [f"{prefix}{i}.{FA_MIXER_ATTR}.{m}" for m in FA_LORA_MODULES]
        else:
            out += [f"{prefix}{i}.{DELTA_MIXER_ATTR}.{m}" for m in DELTA_LORA_MODULES]
    return out


def expected_counts(text_cfg: Mapping, rank: int = LORA_RANK) -> dict:
    """LoRA parameter arithmetic r*(in+out) from text-config dimensions (section 6.1)."""
    h = text_cfg["hidden_size"]
    q_heads, kv_heads, hd = text_cfg["num_attention_heads"], text_cfg["num_key_value_heads"], text_cfg["head_dim"]
    q_out = q_heads * hd * (2 if text_cfg.get("attn_output_gate", True) else 1)  # q_proj also emits the gate
    fa = rank * ((h + q_out) + 2 * (h + kv_heads * hd) + (q_heads * hd + h))
    key_dim = text_cfg["linear_num_key_heads"] * text_cfg["linear_key_head_dim"]
    vh = text_cfg["linear_num_value_heads"]
    value_dim = vh * text_cfg["linear_value_head_dim"]
    d = rank * ((h + 2 * key_dim + value_dim) + (h + value_dim) + 2 * (h + vh) + (value_dim + h))
    types = list(text_cfg["layer_types"])
    n_fa = sum(t == "full_attention" for t in types)
    n_d = sum(t == "linear_attention" for t in types)
    return {"per_fa_layer": fa, "per_delta_layer": d, "fa_layers": n_fa, "delta_layers": n_d,
            "total": fa * n_fa + d * n_d}


def production_expectation(model_cfg: Mapping) -> dict:
    """Derived count from the pinned config must equal the locked constants exactly."""
    counts = expected_counts(model_cfg["text_config"])
    want = {"per_fa_layer": EXPECTED_LORA_PARAMS_PER_FA_LAYER, "per_delta_layer": EXPECTED_LORA_PARAMS_PER_DELTA_LAYER,
            "total": EXPECTED_LORA_PARAMS_TOTAL}
    bad = {k: (counts[k], v) for k, v in want.items() if counts[k] != v}
    if bad:
        raise InventoryError(f"derived LoRA counts differ from the locked specification (derived, locked): {bad}")
    return counts


def enumerate_targets(model, num_layers: int = NUM_LAYERS, fa_layers: Sequence[int] = FULL_ATTENTION_LAYERS,
                      prefix: str = TEXT_LAYERS_PREFIX) -> list[str]:
    """Assert the actual module tree has exactly the expected split projections, all nn.Linear,
    all inside the text decoder; stop on fused projections or wrong mixer families."""
    import torch.nn as nn

    modules = dict(model.named_modules())
    fused = [n for n in modules if n.startswith(prefix) and n.rsplit(".", 1)[-1] in FUSED_DELTA_NAMES]
    if fused:
        raise InventoryError(f"fused Delta projections present {fused[:4]}: a different adapter condition; stop")
    rx = re.compile(target_regex(prefix))
    matched = sorted(n for n in modules if rx.fullmatch(n))
    expected = expected_target_names(num_layers, fa_layers, prefix)
    if matched != sorted(expected):
        raise InventoryError(f"LoRA target inventory mismatch: missing {sorted(set(expected) - set(matched))[:6]}, "
                             f"unexpected {sorted(set(matched) - set(expected))[:6]}")
    for i in range(num_layers):
        wrong = DELTA_MIXER_ATTR if i in fa_layers else FA_MIXER_ATTR
        if f"{prefix}{i}.{wrong}" in modules:
            raise InventoryError(f"layer {i} has a {wrong} mixer, contradicting the pinned layer types")
    for n in expected:
        if not isinstance(modules[n], nn.Linear):
            raise InventoryError(f"{n} is {type(modules[n]).__name__}, expected nn.Linear")
    return expected


def apply_lora(model, seed: int, *, num_layers: int = NUM_LAYERS, fa_layers: Sequence[int] = FULL_ATTENTION_LAYERS,
               prefix: str = TEXT_LAYERS_PREFIX, expected_total: int = EXPECTED_LORA_PARAMS_TOTAL):
    """Freeze everything, seed, wrap with PEFT LoRA, upcast adapters to FP32, verify inventory."""
    import torch
    from peft import LoraConfig, get_peft_model

    targets = enumerate_targets(model, num_layers, fa_layers, prefix)
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                     target_modules=target_regex(prefix), init_lora_weights=True)
    peft_model = get_peft_model(model, cfg)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if ".lora_" in n:
                p.data = p.data.float()
    inv = lora_inventory(peft_model, targets, expected_total)
    inv["init_sha256"] = adapter_sha256(adapter_tensors(peft_model))
    inv["seed"] = seed
    return peft_model, inv


def lora_inventory(model, target_names: Sequence[str], expected_total: int) -> dict:
    import torch

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    targets = set(target_names)
    seen: Counter = Counter()
    for n, p in trainable:
        m = _LORA_PARAM_RE.match(n)
        if not m or m.group("mod") not in targets:
            raise InventoryError(f"trainable parameter outside the LoRA scope: {n}")
        if p.dtype != torch.float32:
            raise InventoryError(f"{n} is {p.dtype}; trainable LoRA must be FP32")
        seen[(m.group("mod"), m.group("ab"))] += 1
    for t in targets:
        if seen[(t, "A")] != 1 or seen[(t, "B")] != 1:
            raise InventoryError(f"{t} lacks exactly one trainable lora_A and lora_B")
    frozen_lora = [n for n, p in model.named_parameters() if ".lora_" in n and not p.requires_grad]
    if frozen_lora:
        raise InventoryError(f"frozen LoRA tensors present: {frozen_lora[:4]}")
    total = sum(p.numel() for _, p in trainable)
    if total != expected_total:
        raise InventoryError(f"trainable LoRA parameters {total:,} != expected {expected_total:,}")
    frozen_dtypes = Counter(str(p.dtype) for _, p in model.named_parameters() if not p.requires_grad)
    return {"trainable_params": total, "trainable_tensors": len(trainable), "target_modules": len(targets),
            "frozen_dtype_tensor_counts": dict(frozen_dtypes)}


def adapter_tensors(model) -> dict:
    """Trainable adapter tensors by parameter name (detached CPU copies, dtype preserved)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def adapter_sha256(tensors: Mapping) -> str:
    import hashlib

    import torch

    h = hashlib.sha256()
    for n in sorted(tensors):
        t = tensors[n].detach().cpu().contiguous()
        h.update(f"{n}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_adapter_tensors(model, tensors: Mapping) -> None:
    """Copy saved FP32 adapter tensors into the trainable parameters (exact name/shape/dtype)."""
    import torch

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(tensors):
        raise InventoryError("checkpoint adapter names differ from the model's trainable parameters")
    with torch.no_grad():
        for n, p in params.items():
            t = tensors[n]
            if t.dtype != p.dtype or tuple(t.shape) != tuple(p.shape):
                raise InventoryError(f"{n}: checkpoint {t.dtype}{tuple(t.shape)} vs model {p.dtype}{tuple(p.shape)}")
            p.copy_(t.to(p.device))
