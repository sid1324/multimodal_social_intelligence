"""Bounded Qwen3.5 native training bridges, attributed to pinned Transformers.

Preprocessing follows modeling_qwen3_5.py at the immutable runtime pin. No
process-global function or backend registry is modified. Masks are alignment
inputs, never inferred here. Keep the context open through loss.backward().
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from grounded_norm_sft.arms import require_trainable_arm
from grounded_norm_sft.runtime import pins
from .endpoints import delta_endpoint, full_attention_endpoint, _masks
from .scope import one_site, dispatch


def _unsupported(kwargs):
    if any(kwargs.get(k) is not None for k in (
        "cache_params", "past_key_values", "past_key_value", "cu_seq_lens_q",
        "cu_seq_lens_k", "cu_seqlens", "cache_position")) or kwargs.get("use_cache", False):
        raise ValueError("SRER training adapters require uncached, unpacked full sequences")


def _delta(module, native, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    _unsupported(dict(kwargs, cache_params=cache_params))
    hidden_states = native.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    mixed = module.in_proj_qkv(hidden_states).transpose(1, 2)
    z = module.in_proj_z(hidden_states).reshape(batch, length, -1, module.head_v_dim)
    b, a = module.in_proj_b(hidden_states), module.in_proj_a(hidden_states)
    mixed = native.causal_conv1d_fn(mixed, module.conv1d.weight.squeeze(1),
                                   module.conv1d.bias, activation=module.activation, **kwargs)
    q, k, v = torch.split(mixed.transpose(1, 2),
                          [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = q.reshape(batch, length, -1, module.head_k_dim)
    k = k.reshape(batch, length, -1, module.head_k_dim)
    v = v.reshape(batch, length, -1, module.head_v_dim)
    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    if module.num_v_heads // module.num_k_heads > 1:
        q = q.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
        k = k.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    return q, k, v, g, beta, z


def _attention(module, native, hidden_states, position_embeddings, attention_mask,
               past_key_values=None, **kwargs):
    _unsupported(dict(kwargs, past_key_values=past_key_values))
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, module.head_dim)
    q, gate = torch.chunk(module.q_proj(hidden_states).view(*shape, -1, module.head_dim * 2), 2, -1)
    gate = gate.reshape(*shape, -1)
    q = module.q_norm(q.view(head_shape)).transpose(1, 2)
    k = module.k_norm(module.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
    v = module.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    q, k = native.apply_rotary_pos_emb(q, k, *position_embeddings)
    clean, weights = native.eager_attention_forward(
        module, q, k, v, attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling, **kwargs)
    return clean, weights, v, gate


@contextmanager
def native_site(module, *, arm, source, rows, eta=0.25, enabled=True, diagnostic=None, ordinal=None, protocol=None):
    """Install one native mixer bridge for a forward AND its backward.

    F-T accepts FA only, D-T Delta only; H-T/H-R accept either native family.
    Empty support and eta-zero/disabled preserve the literal native callable.
    Exiting before checkpoint backward is a caller error: recomputation must
    run inside this context. Export and evaluation must run after exit.
    """
    require_trainable_arm(arm)
    if arm not in ("F-T", "D-T", "H-T", "H-R"):
        raise ValueError("only intervention arms may install a native bridge")
    from .eta050 import require_dose
    require_dose(eta, protocol)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    pin = pins.load_pins()[0]
    pins.verify_transformers(pin)
    pins.verify_imported_modeling_module(pin)
    if type(module) is native.Qwen3_5Attention:
        family = "FA"
        if module.config._attn_implementation != "eager":
            raise ValueError("SRER full attention requires the pinned eager backend")
    elif type(module) is native.Qwen3_5GatedDeltaNet:
        family = "D"
    else:
        raise TypeError("expected a pinned native Qwen3.5 text mixer")
    if (arm == "F-T" and family != "FA") or (arm == "D-T" and family != "D"):
        raise ValueError("intervention arm and native mixer family disagree")
    # Freeze caller masks for checkpoint replay (alignment owns their meaning).
    source, rows = source.clone(), rows.clone()
    emitted = False
    if diagnostic is not None and (not isinstance(ordinal, int) or ordinal < 0):
        raise ValueError('diagnostics require a nonnegative forward ordinal')
    def capture(**values):
        nonlocal emitted
        if emitted:
            return
        from .diagnostics import endpoint_record
        record = endpoint_record(site=module.layer_idx, family=family, ordinal=ordinal,
                                 source=source, rows=rows, eta=eta, **values)
        diagnostic(record)
        emitted = True
    def custom(*args, **kwargs):
        if family == "D":
            q, k, v, g, beta, z = _delta(module, native, *args, **kwargs)
            return delta_endpoint(q, k, v, g, beta, z, source=source, rows=rows,
                                  norm=module.norm, out_proj=module.out_proj,
                                  kernel=native.torch_chunk_gated_delta_rule, eta=eta, protocol=protocol,
                                  diagnostic=capture if diagnostic is not None and not emitted else None)
        clean, weights, v, gate = _attention(module, native, *args, **kwargs)
        output = full_attention_endpoint(clean, weights, v, gate, source=source, rows=rows,
                                         num_key_value_groups=module.num_key_value_groups,
                                         repeat_kv=native.repeat_kv, out_proj=module.o_proj, eta=eta, protocol=protocol,
                                         diagnostic=capture if diagnostic is not None and not emitted else None)
        return output, weights
    def bridge(original, *args, **kwargs):
        if enabled and eta != 0:
            hidden = args[0] if args else kwargs['hidden_states']
            _masks(source, rows, *hidden.shape[:2])
            if source.device != hidden.device or rows.device != hidden.device:
                raise ValueError("alignment masks must be on the hidden-state device")
            if not source.any():
                return original(*args, **kwargs)
        return dispatch(original, custom, *args, enabled=enabled, eta=eta, protocol=protocol, **kwargs)
    with one_site(module, bridge, arm=arm):
        yield module


@contextmanager
def diagnostic_eta_one(module, *, source, rows, diagnostic, ordinal=0):
    """Excluded native endpoint anchor only, forbidden with gradients or training.

    The scientific operator remains eta=.25. This diagnostic reuses its exact
    endpoint calculation and selects the excluded endpoint after native transforms.
    """
    if torch.is_grad_enabled() or module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError('eta=1 anchor requires eval, frozen parameters and torch.no_grad')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    pin = pins.load_pins()[0]
    pins.verify_transformers(pin)
    pins.verify_imported_modeling_module(pin)
    if type(module) is native.Qwen3_5Attention:
        family = 'FA'
        if module.config._attn_implementation != 'eager':
            raise ValueError('diagnostic full attention requires eager native backend')
    elif type(module) is native.Qwen3_5GatedDeltaNet:
        family = 'D'
    else:
        raise TypeError('expected pinned native text mixer')
    source, rows = source.clone(), rows.clone()
    def bridge(original, *args, **kwargs):
        if torch.is_grad_enabled() or module.training:
            raise ValueError('eta=1 diagnostic cannot be used in training')
        hidden = args[0] if args else kwargs['hidden_states']
        _masks(source, rows, *hidden.shape[:2])
        if source.device != hidden.device or rows.device != hidden.device or not source.any():
            raise ValueError('nonempty masks on native device required')
        values = {}
        if family == 'D':
            q, k, v, g, beta, z = _delta(module, native, *args, **kwargs)
            delta_endpoint(q, k, v, g, beta, z, source=source, rows=rows,
                norm=module.norm, out_proj=module.out_proj,
                kernel=native.torch_chunk_gated_delta_rule, eta=.25,
                diagnostic=lambda **data: values.update(data))
        else:
            clean, weights, v, gate = _attention(module, native, *args, **kwargs)
            full_attention_endpoint(clean, weights, v, gate, source=source, rows=rows,
                num_key_value_groups=module.num_key_value_groups, repeat_kv=native.repeat_kv,
                out_proj=module.o_proj, eta=.25, diagnostic=lambda **data: values.update(data))
        output = torch.where(rows[..., None], values['excluded_output'], values['clean_output'])
        values['output'] = output
        from .diagnostics import endpoint_record
        record = endpoint_record(site=module.layer_idx, family=family, ordinal=ordinal,
                                 source=source, rows=rows, eta=1., **values)
        record['eta'] = 1.
        diagnostic(record)
        return output if family == 'D' else (output, weights)
    with one_site(module, bridge, arm='H-T'):
        yield module
