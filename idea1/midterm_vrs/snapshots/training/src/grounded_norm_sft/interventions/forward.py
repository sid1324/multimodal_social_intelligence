"""Supplied-model native forwards and immutable cache validation; no model loading."""
from contextlib import contextmanager
from copy import deepcopy
import torch
from .cache import provenance, read_record, digest
from .data import ManifestInputs, fingerprint
from ..runtime.model import NativeForward as CleanNativeForward, set_train_mode
from .native_adapters import native_site


class NativeForward:
    def __init__(self, model, *, alignment_ids, diagnostic=None):
        self.model, self.alignment_ids = model, dict(alignment_ids)
        self.diagnostic, self._diagnostic_ordinal = diagnostic, 0
        self._bound = {}
        self._streams = []
        self.native = CleanNativeForward(model, str(next(model.parameters()).device))
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        self.sites = {}
        for module in model.modules():
            if type(module) in (native.Qwen3_5Attention, native.Qwen3_5GatedDeltaNet):
                layer = module.layer_idx
                if layer in self.sites:
                    raise ValueError('duplicate text mixer layer')
                self.sites[layer] = module
        if set(self.sites) != set(range(32)):
            raise ValueError('all 32 native text mixers required')
        config = model.config
        for field, key in (('image_token_id', 'image_token_id'), ('vision_start_token_id', 'vision_start_id'),
                           ('vision_end_token_id', 'vision_end_id')):
            if getattr(config, field) != self.alignment_ids[key]:
                raise ValueError('model/processor token IDs mismatch')
        if config.text_config._attn_implementation != 'eager' or config.vision_config._attn_implementation != 'eager':
            raise ValueError('native eager text and vision backends required')

    def bind(self, serialized):
        if isinstance(serialized, ManifestInputs):
            if any(stream.task == serialized.task for stream in self._streams):
                raise ValueError('only one streamed mapping per task may be bound')
            self._streams.append(serialized)
            return
        for item in serialized.values():
            self._bound[id(item)] = self._fingerprint(item)

    @staticmethod
    def _fingerprint(item):
        return fingerprint(item)

    def __call__(self, task, serialized):
        if id(serialized) in self._bound and self._fingerprint(serialized) != self._bound[id(serialized)]:
            raise ValueError('bound serialized input changed after cache validation')
        for stream in self._streams:
            if stream.task == task:
                stream.verify_item(serialized)
        if serialized.task != task:
            raise ValueError('serialized task mismatch')
        if any(k in serialized.model_inputs for k in ('labels', 'label', 'past_key_values', 'inputs_embeds', 'use_cache')):
            raise ValueError('processor inputs must contain no labels or caches')
        device = next(self.model.parameters()).device
        inputs = {k: torch.as_tensor(v, device=device) for k, v in serialized.model_inputs.items()}
        if inputs['input_ids'].ndim == 1:
            inputs['input_ids'] = inputs['input_ids'][None]
        if inputs['input_ids'].shape != (1, len(serialized.input_ids)):
            raise ValueError('one unpacked complete sequence required')
        if inputs['input_ids'][0].tolist() != serialized.input_ids:
            raise ValueError('serialized token IDs mismatch')
        from dataclasses import replace
        # Reuse B1's frozen eval/no-grad visual encoder and cache-free native execution.
        if self.model.training:
            set_train_mode(self.model)
        tensor_serialized = replace(serialized, model_inputs=inputs)
        logits = self.native(tensor_serialized, grad=torch.is_grad_enabled())
        return logits, serialized.answer_token_ids

    @contextmanager
    def perturbed(self, serialized, support, row, *, arm, eta, protocol=None):
        device = next(self.model.parameters()).device
        source = torch.zeros((1, len(serialized.input_ids)), dtype=torch.bool, device=device)
        rows = torch.zeros_like(source)
        selected = support['random' if arm == 'H-R' else 'targeted']
        source[0, [support['patches'][i] for i in selected]] = True
        rows[0, support['rows']] = True
        ordinal = self._diagnostic_ordinal
        self._diagnostic_ordinal += 1
        with native_site(self.sites[row['site_layer']], arm=arm, source=source, rows=rows, eta=eta,
                         diagnostic=self.diagnostic, ordinal=ordinal, protocol=protocol):
            yield


def validated_supports(serialized, cache_paths, *, pins, image_sha256, alignment_ids):
    """Validate every current serialized record before the first optimizer update."""
    records = {}
    for key, item in serialized.items():
        if key != item.example_id:
            raise ValueError('serialized example ID mismatch')
        if any(k in item.model_inputs for k in ('labels', 'label')):
            raise ValueError('support inputs cannot contain labels')
        expected = provenance(item, **dict(pins, image_sha256=image_sha256[key]), **alignment_ids)
        records[key] = deepcopy(read_record(cache_paths[key], expected_provenance=expected))
    return records
