"""Frozen native FA19 acquisition and immutable action/generic support caches.

Callers supply already audited base weights and processor; hashes identify their
artifacts and do not substitute for a weight-file audit. Nothing is loaded here.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import json

import torch

from ..constants import SELECTOR_LAYER, FA_NUM_Q_HEADS
from ..runtime import pins as runtime_pins
from .alignment import align, AlignmentError
from .cache import build_record, write_record, read_record, provenance, digest
from .selection import SystematicAlignmentFailure


@dataclass(frozen=True)
class Reference:
    probabilities: torch.Tensor
    diagnostics: dict


def _validate(model):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    pin = runtime_pins.load_pins()[0]
    runtime_pins.verify_transformers(pin)
    runtime_pins.verify_imported_modeling_module(pin)
    if type(model) is not native.Qwen3_5ForConditionalGeneration:
        raise TypeError('reference requires an unwrapped pinned Qwen3.5 base model')
    if any('forward' in m.__dict__ for m in model.modules()):
        raise ValueError('reference requires native forwards without scoped interventions')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('reference base parameters must already be frozen')
    if any('lora' in name.lower() or 'adapter' in name.lower() for name, _ in model.named_modules()):
        raise ValueError('reference must contain no LoRA/adapters')
    modules = [m for m in model.modules() if type(m) is native.Qwen3_5Attention and m.layer_idx == SELECTOR_LAYER]
    if len(modules) != 1:
        raise ValueError('exactly one native full-attention layer 19 required')
    module = modules[0]
    if module.config.num_hidden_layers != 32 or module.config.num_attention_heads != FA_NUM_Q_HEADS:
        raise ValueError('reference requires 32 layers and all 16 query heads')
    if module.config._attn_implementation != 'eager':
        raise ValueError('reference acquisition requires supplied native eager backend')
    if getattr(model, '_srer_reference_active', False):
        raise RuntimeError('one reference acquisition per model instance')
    return module


def acquire(model, serialized, *, alignment_ids):
    """Run the exact serialized inputs, eval/no-grad, returning native completed-row P.

    Hooks observe native outputs only. Training flags and both hook handles are
    restored even on failure. No cache, labels, adapter or model update is used.
    """
    module = _validate(model)
    if any(k in serialized.model_inputs for k in ('labels', 'label', 'past_key_values', 'inputs_embeds', 'use_cache')):
        raise ValueError('reference accepts processor inputs only, without labels or caches')
    if any(k not in serialized.model_inputs for k in ('pixel_values', 'image_grid_thw', 'mm_token_type_ids')):
        raise ValueError('native multimodal processor inputs required')
    for field, key in (('image_token_id', 'image_token_id'), ('vision_start_token_id', 'vision_start_id'), ('vision_end_token_id', 'vision_end_id')):
        if getattr(model.config, field) != alignment_ids[key]:
            raise ValueError('model/processor token IDs mismatch')
    a = align(serialized, **alignment_ids)
    row = serialized.completed_action_row
    captured = {}
    def gate_hook(_, args, output):
        if torch.is_grad_enabled():
            raise RuntimeError('reference requires no-grad')
        if output.shape[:2] != (1, len(serialized.input_ids)):
            raise ValueError('reference requires one unpacked complete sequence')
        gate = output.view(1, output.shape[1], FA_NUM_Q_HEADS, 2 * module.head_dim).chunk(2, -1)[1]
        captured['gate'] = gate[0, row].sigmoid().float().mean(-1).cpu()
    def attention_hook(_, args, output):
        p = output[1]
        if p is None or p.shape != (1, FA_NUM_Q_HEADS, len(serialized.input_ids), len(serialized.input_ids)):
            raise ValueError('native full attention probabilities unavailable')
        captured['p'] = p[0, :, row].detach().cpu().clone()
    modes = [(m, m.training) for m in model.modules()]
    model._srer_reference_active = True
    try:
        model.eval()
        with ExitStack() as stack:
            stack.callback(module.q_proj.register_forward_hook(gate_hook).remove)
            stack.callback(module.register_forward_hook(attention_hook).remove)
            device = next(model.parameters()).device
            inputs = {k: (v.to(device) if hasattr(v, 'to') else torch.tensor(v, device=device))
                      for k, v in serialized.model_inputs.items()}
            if 'input_ids' in inputs and inputs['input_ids'].ndim == 1:
                inputs['input_ids'] = inputs['input_ids'][None]
            with torch.no_grad():
                model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    finally:
        for m, training in modes:
            m.training = training
        del model._srer_reference_active
    if set(captured) != {'p', 'gate'}:
        raise RuntimeError('reference layer was not executed exactly as expected')
    p, gate = captured['p'], captured['gate']
    if not torch.isfinite(gate).all():
        # Existing selection policy marks nonfinite reference values clean/unsupported.
        p = p.clone().fill_(float('nan'))
    boundaries = (a.patches[0] - 1, a.patches[-1] + 1)
    # Explicit coordinates, not a claim that these positions are proven sinks.
    adjacent = tuple(sorted({i for b in (0, *boundaries) for i in (b-1, b+1)
                             if 0 <= i <= row and i not in a.patches and i not in boundaries}))
    bos = getattr(model.config.text_config, 'bos_token_id', None)
    positions = {'bos': tuple(i for i, token in enumerate(serialized.input_ids) if token == bos and i <= row),
                 'prompt_start': (0,), 'vision_boundaries': boundaries, 'sink_adjacent': adjacent,
                 'first_two_patches': a.patches[:2]}
    diagnostics = {'completed_action_row': row, 'gate_mean_per_head': gate.tolist(),
                   'positions': {k: list(v) for k, v in positions.items()},
                   'attention_per_head': {k: p[:, list(v)].float().tolist() for k, v in positions.items()},
                   'patch_mass_per_head': p[:, list(a.patches)].float().sum(-1).tolist()}
    return Reference(p, diagnostics)


class ReferencePipeline:
    """Input-only action/generic acquisition, with durable generic reuse per image.

    Supply model/processor/template SHA256 pins at construction and image SHA256
    per call. Cache reads always validate full expected provenance. Diagnostics
    are create-only sidecars tied to the exact support record checksum.
    """
    def __init__(self, model, serializer, directory, *, pins, alignment_ids):
        _validate(model)
        self.model, self.serializer = model, serializer
        self.directory, self.pins, self.alignment_ids = Path(directory), dict(pins), dict(alignment_ids)
        self._generic = {}
        self.counts = {'total': 0, 'supported': 0, 'unsupported_reasons': {}}

    def _one(self, serialized, image_sha256):
        pins = dict(self.pins, image_sha256=image_sha256)
        identity = provenance(serialized, **pins, **self.alignment_ids)
        path = self.directory / (digest(identity) + '.json')
        diagnostics_path = path.with_suffix('.diagnostics.json')
        if path.exists():
            record = read_record(path, expected_provenance=identity)
            if not diagnostics_path.exists():
                raise ValueError('reference cache diagnostics missing')
            sidecar = json.loads(diagnostics_path.read_bytes())
            if sidecar['record_sha256'] != record['sha256'] or digest(sidecar['diagnostics']) != sidecar['sha256']:
                raise ValueError('reference diagnostics checksum mismatch')
            return dict(path=path, record=record, diagnostics_path=diagnostics_path,
                        diagnostics=sidecar['diagnostics'], reused=True)
        try:
            align(serialized, **self.alignment_ids)
        except AlignmentError:
            reference = Reference(torch.empty(0), {'alignment_unsupported': True})
        else:
            reference = acquire(self.model, serialized, alignment_ids=self.alignment_ids)
        record = build_record(serialized, reference.probabilities, pins=pins, alignment_ids=self.alignment_ids)
        # JSON cannot encode NaN: diagnostics explicitly record finite failure.
        diagnostics = reference.diagnostics
        if not torch.isfinite(reference.probabilities).all():
            diagnostics = {'nonfinite_reference': True}
        from ..hashing import canonical_json_bytes
        sidecar = dict(record_sha256=record['sha256'], diagnostics=diagnostics, sha256=digest(diagnostics))
        self.directory.mkdir(parents=True, exist_ok=True)
        raw = canonical_json_bytes(sidecar)
        try:
            with diagnostics_path.open('xb') as stream:
                stream.write(raw)
        except FileExistsError:
            if diagnostics_path.read_bytes() != raw:
                raise ValueError('immutable reference diagnostics conflict')
        write_record(self.directory, record)
        return dict(path=path, record=record, diagnostics_path=diagnostics_path,
                    diagnostics=diagnostics, reused=False)

    def run(self, serialized, *, image_id, image, image_sha256, systemic_alignment_failure=False):
        _validate(self.model)
        if any(k in serialized.model_inputs for k in ('labels', 'label', 'past_key_values', 'inputs_embeds', 'use_cache')):
            raise ValueError('reference accepts processor inputs only, without labels or caches')
        if systemic_alignment_failure:
            raise SystematicAlignmentFailure('systematic alignment failure; stop per frozen 5.2')
        # Serialize on every request to validate the same processor/prompt/grid;
        # actual model acquisition is cached once per immutable image identity.
        generic = self.serializer.selector_generic(image_id, image)
        key = (str(image_id), image_sha256)
        identity = provenance(generic, **dict(self.pins, image_sha256=image_sha256), **self.alignment_ids)
        if key in self._generic and self._generic[key] != identity:
            raise ValueError('generic image cache provenance mismatch')
        try:
            align(serialized, **self.alignment_ids)
        except AlignmentError:
            image_fields = ('pixel_values',)
        else:
            image_fields = ('image_grid_thw', 'pixel_values')
        for field in image_fields:
            left, right = serialized.model_inputs.get(field), generic.model_inputs.get(field)
            if left is None or right is None or not torch.equal(torch.as_tensor(left), torch.as_tensor(right)):
                raise ValueError('action/generic processed image mismatch')
        self._generic[key] = identity
        action = self._one(serialized, image_sha256)
        generic_result = self._one(generic, image_sha256)
        support = action['record']['payload']['support']
        self.counts['total'] += 1
        self.counts['supported'] += int(support['supported'])
        if not support['supported']:
            reasons = self.counts['unsupported_reasons']
            reasons[support['reason']] = reasons.get(support['reason'], 0) + 1
        targeted = support['targeted']
        return dict(action=action, generic=generic_result, coverage=dict(
            total=self.counts['total'], supported=self.counts['supported'],
            unsupported_reasons=dict(self.counts['unsupported_reasons'])),
            first_two_patch_inclusion=[i in targeted for i in (0, 1)])
