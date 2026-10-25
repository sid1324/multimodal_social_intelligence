"""Native-interface SRER operators, plan v1.2 section 5.

Inputs are live native tensors after convolution/RoPE and coefficient computation.
No cache, packed sequence, or incremental decoding is supported by this slice.
"""
from collections.abc import Callable

import torch


def _eta(eta, protocol=None):
    from .eta050 import require_dose
    require_dose(eta, protocol)


def _masks(source, rows, batch, length):
    for name, mask in (("source", source), ("rows", rows)):
        if mask.dtype != torch.bool or mask.shape != (batch, length):
            raise ValueError(f"{name} must be a [batch, sequence] boolean tensor")
    if (source & rows).any():
        raise ValueError("source positions cannot be intervention rows")


def interpolate(clean, excluded, rows, eta=0.25, *, protocol=None):
    """FP32 endpoint mixing, with literal identity for disabled operation."""
    _eta(eta, protocol)
    if eta == 0:
        return clean
    if clean.shape != excluded.shape or rows.shape != clean.shape[:2]:
        raise ValueError("endpoint/row shape mismatch")
    if not torch.isfinite(clean).all() or not torch.isfinite(excluded).all():
        raise FloatingPointError("nonfinite native endpoint")
    mixed = (clean.float() + eta * (excluded.float() - clean.float())).to(clean.dtype)
    return torch.where(rows[..., None], mixed, clean)


def delta_endpoint(query, key, value, g, beta, z, *, source, rows,
                   norm, out_proj, kernel: Callable, eta=0.25, diagnostic=None, protocol=None):
    """Call the native chunk recurrence twice, sharing all live coefficients.

    `kernel` must be the pinned torch_chunk_gated_delta_rule. `norm` is the
    actual Qwen3_5RMSNormGated module, preserving epsilon, dtype and SiLU order.
    Returns projected output; initial state is deliberately fixed to None.
    """
    _eta(eta, protocol)
    batch, length, _, width = value.shape
    _masks(source, rows, batch, length)

    def scan(v):
        return kernel(query, key, v, g=g, beta=beta, initial_state=None,
                      output_final_state=False, use_qk_l2norm_in_kernel=True)[0]

    def transform(y):
        return out_proj(norm(y.reshape(-1, width), z.reshape(-1, width))
                        .reshape(batch, length, -1))

    y = scan(value)
    clean = transform(y)
    if eta == 0 or not source.any():
        return clean
    excluded_y = scan(value.masked_fill(source[..., None, None], 0))
    excluded = transform(excluded_y)
    output = interpolate(clean, excluded, rows, eta, protocol=protocol)
    if diagnostic is not None:
        with torch.no_grad():
            def normalized(x):
                x = x.detach()
                f = x.float()
                return norm.weight.detach() * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + norm.variance_epsilon)).to(x.dtype)
            diagnostic(clean_readout=y.detach(), excluded_readout=excluded_y.detach(),
                       clean_normalized=normalized(y), excluded_normalized=normalized(excluded_y),
                       clean_postgate=norm(y.detach().reshape(-1, width), z.detach().reshape(-1, width)).reshape_as(y),
                       excluded_postgate=norm(excluded_y.detach().reshape(-1, width), z.detach().reshape(-1, width)).reshape_as(excluded_y),
                       clean_output=clean.detach(), excluded_output=excluded.detach(), output=output.detach(),
                       native_epsilon=norm.variance_epsilon)
    return output


def full_attention_endpoint(clean_readout, coefficients, value, gate, *,
                            source, rows, num_key_value_groups, repeat_kv,
                            out_proj, eta=0.25, diagnostic=None, protocol=None):
    """Use native eager probabilities (including dropout) and live native gate.

    clean_readout is the native eager result [B,T,H,D]; value is [B,KV,T,D].
    No softmax recomputation or coefficient renormalization is performed.
    """
    _eta(eta, protocol)
    batch, length = clean_readout.shape[:2]
    _masks(source, rows, batch, length)

    def transform(y):
        return out_proj(y.reshape(batch, length, -1).contiguous() * gate.sigmoid())

    clean = transform(clean_readout)
    if eta == 0 or not source.any():
        return clean
    excluded_values = value.masked_fill(source[:, None, :, None], 0)
    expanded = repeat_kv(excluded_values, num_key_value_groups)
    excluded_readout = (coefficients @ expanded).transpose(1, 2).contiguous()
    excluded = transform(excluded_readout)
    output = interpolate(clean, excluded, rows, eta, protocol=protocol)
    if diagnostic is not None:
        with torch.no_grad():
            gate_value = gate.detach().sigmoid().reshape_as(clean_readout)
            diagnostic(clean_readout=clean_readout.detach(), excluded_readout=excluded_readout.detach(),
                       clean_postgate=clean_readout.detach()*gate_value,
                       excluded_postgate=excluded_readout.detach()*gate_value,
                       clean_output=clean.detach(), excluded_output=excluded.detach(), output=output.detach())
    return output
