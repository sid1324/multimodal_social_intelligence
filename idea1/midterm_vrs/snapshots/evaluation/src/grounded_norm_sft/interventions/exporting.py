"""Bounded offline standard-native PEFT export proof; no training entry point."""
from pathlib import Path
import json
import math
import os
import subprocess
import shutil
import sys
import tempfile
import time

import torch
from peft import get_peft_model_state_dict

from ..runtime.errors import ExportError
from ..runtime.export import (assert_native_source, assert_adapter_equal, native_repeat_floor,
                              export_tolerance, compare_to_reference)

# Deliberately self-contained: this source executes under python -I, with no project import.
_NATIVE = '''
import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from peft import PeftModel, get_peft_model_state_dict

def infer(model, fixtures, guard=None):
    model.eval()
    base = model.get_base_model()
    outputs = {}
    with torch.no_grad():
        for key, item in fixtures.items():
            if guard is not None:
                guard()
            inputs = {k: v.to(next(model.parameters()).device) for k, v in item['inputs'].items()}
            base.model.rope_deltas = None
            pixels = inputs.pop('pixel_values', None)
            mm = None
            if pixels is not None:
                mm = {'image': base.model.get_image_features(pixels, inputs['image_grid_thw'], return_dict=True)}
            logits = model(**inputs, mm_encoder_outputs=mm, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            outputs[key] = {'full_logprobs': logits.log_softmax(-1)[item['answer_ids']].tolist(),
                            'argmax_token_id': int(logits.argmax())}
    return outputs
'''
_CHILD = _NATIVE + '''
import sys
request, destination = sys.argv[1:]
torch.set_num_threads(1)
r = torch.load(request, weights_only=True)
base = Qwen3_5ForConditionalGeneration.from_pretrained(r['base'], local_files_only=True,
    trust_remote_code=False, attn_implementation='eager', dtype=r['dtype']).to(r['device'])
assert str(next(base.parameters()).dtype) == r['source_dtype'], 'base dtype policy mismatch'
model = PeftModel.from_pretrained(base, r['adapter'], local_files_only=True,
    # Promote adapter slots before loading FP32 weights; BF16 slots would round them on copy.
    is_trainable=False, autocast_adapter_dtype=True)
for name, module in model.named_modules():
    assert 'forward' not in vars(module), name
    assert not any(getattr(module, key, {}) for key in
        ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')), name
assert native.causal_conv1d_fn.__module__ == native.__name__
import importlib.util
assert not any(importlib.util.find_spec(k) for k in ('fla', 'causal_conv1d', 'kernels'))
out = infer(model, r['fixtures'])
assert not any(k.startswith('grounded_norm_sft') for k in sys.modules)
torch.save({'outputs': out, 'adapter': get_peft_model_state_dict(model),
            'pid': __import__('os').getpid(), 'project_imports': [],
            'delta_backend': 'torch_native_fallback'}, destination)
'''


def _source_devices(model):
    return sorted({str(t.device) for t in list(model.parameters()) + list(model.buffers())})


def _relocate_source(model, callback, *, exclusive_gpu):
    if callback is not None:
        callback(model)
    devices = _source_devices(model)
    if exclusive_gpu and any(torch.device(d).type != 'cpu' for d in devices):
        raise ExportError('exclusive GPU export requires all source parameters/buffers off CUDA and on CPU')
    return devices


def export_native_adapter(model, destination, *, base_dir, processor, fixtures, identity, repeats=10,
                          timeout=120, child_device='cpu', base_dtype='auto', delta_backend='torch_native_fallback',
                          source_relocator=None, exclusive_gpu=True, processor_dir=None, fixture_only=False, deadline=None):
    """Save an unmerged adapter and prove two isolated local-only native reloads.

    Fixtures map stable IDs to serialized input-only records. ``base_dir`` must
    contain pristine native weights saved before PEFT wrapping. Identity binds
    this proof to the caller's immutable protocol/model/processor/template pins.
    Production copies the verified original seven processor assets byte-for-byte;
    fixture_only permits a synthetic CPU processor save and makes no asset claim.
    source_relocator(model) runs after native evidence/saves and before child loads.
    Its device changes persist even on failure; training/eval flags are restored.
    Exclusive GPU policy verifies source parameters and buffers have left CUDA;
    callers must also release optimizer state and any external CUDA references.
    """
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ExportError('finite monotonic overall deadline required')
    def check_budget():
        if deadline is not None and time.monotonic() >= deadline:
            from .pipeline import BudgetExpired
            raise BudgetExpired('export overall walltime exhausted; incomplete')
    check_budget()
    # Relocation is persistent and explicit; caller owns optimizer/state references.
    # Never recompute the reference after moving the source to another backend.
    if type(exclusive_gpu) is not bool or type(fixture_only) is not bool:
        raise ExportError('explicit boolean lifecycle/fixture policy required')
    if child_device.startswith('cuda') and not exclusive_gpu:
        raise ExportError('CUDA children require exclusive_gpu policy')
    if fixture_only and child_device != 'cpu':
        raise ExportError('synthetic processor fixtures are CPU-only')
    if delta_backend != 'torch_native_fallback' or base_dtype not in ('auto', 'float32', 'bfloat16'):
        raise ExportError('unsupported explicit native backend/dtype policy')
    if child_device != 'cpu' and not child_device.startswith('cuda'):
        raise ExportError('explicit CPU or CUDA child device required')
    from ..runtime.model import delta_dispatch_inventory
    delta_dispatch_inventory()
    if not identity or any(not isinstance(v, str) or not v.strip() for v in identity.values()):
        raise ExportError('nonempty immutable export identities required')
    if not fixtures or any(not isinstance(k, str) or not k for k in fixtures):
        raise ExportError('nonempty fixed fixture IDs required')
    if type(repeats) is not int or repeats < 10:
        raise ExportError('at least ten native repeats required')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ExportError('finite positive child timeout required')
    base_dir = Path(base_dir).resolve()
    if not (base_dir / 'config.json').is_file():
        raise ExportError('pristine local native base is missing')
    if not {'protocol', 'model', 'processor', 'template'} <= set(identity):
        raise ExportError('protocol/model/processor/template identities required')
    from ..runtime.pins import processor_assets_digest
    processor_source = None
    processor_digest = None
    if not fixture_only:
        if processor_dir is None:
            raise ExportError('verified original processor_dir required')
        processor_source = Path(processor_dir).resolve()
        processor_digest, processor_files = processor_assets_digest(processor_source)
        if processor_digest != identity['processor']:
            raise ExportError('original processor digest differs from export identity')
    elif processor_dir is not None:
        raise ExportError('fixture-only processor cannot use production assets')
    check_budget()
    inventory = assert_native_source(model)
    if inventory['hooks']:
        raise ExportError('native export requires no hooks')
    if any(getattr(c, '_attn_implementation', None) != 'eager' for c in
           (model.config.text_config, model.config.vision_config)):
        raise ExportError('native export requires eager text and vision')
    packed = {}
    answer_count = None
    for key, ser in fixtures.items():
        if key != ser.example_id or ser.final_prompt_row != len(ser.input_ids) - 1:
            raise ExportError('fixture identity/final prompt mismatch')
        if any(k in ser.model_inputs for k in ('labels', 'label', 'past_key_values', 'inputs_embeds', 'use_cache', 'mm_encoder_outputs')):
            raise ExportError('fixtures must be input-only and cache-free')
        ids = list(ser.answer_token_ids)
        eos = model.config.text_config.eos_token_id
        excluded = set(eos if isinstance(eos, list) else [eos])
        excluded.add(getattr(processor.tokenizer, 'eos_token_id', None))
        excluded.add(getattr(processor.tokenizer, 'pad_token_id', None))
        if not ids or len(set(ids)) != len(ids) or any(type(x) is not int or x < 0 or x >= model.config.text_config.vocab_size for x in ids) or set(ids) & excluded:
            raise ExportError('invalid answer IDs or EOS supervision')
        answer_count = answer_count or len(ids)
        if len(ids) != answer_count:
            raise ExportError('fixture answer lengths differ')
        inputs = {k: torch.as_tensor(v).detach().cpu().clone() for k, v in ser.model_inputs.items()}
        if inputs['input_ids'].shape != (1, len(ser.input_ids)) or inputs['input_ids'][0].tolist() != ser.input_ids:
            raise ExportError('fixture input IDs mismatch')
        if any(not torch.isfinite(v).all() for v in inputs.values()):
            raise ExportError('nonfinite fixture')
        packed[key] = dict(inputs=inputs, answer_ids=ids)
    saved = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    if any(not torch.isfinite(v).all() for v in saved.values()):
        raise ExportError('nonfinite adapter')
    assert_adapter_equal(saved, saved)
    namespace = {}
    exec(_NATIVE, namespace)
    reference_devices = _source_devices(model)
    source_dtype = str(next(model.get_base_model().parameters()).dtype)
    training_modes = [(module, module.training) for module in model.modules()]
    try:
        def infer_bounded(m):
            check_budget()
            result = namespace['infer'](m, packed, guard=check_budget)
            check_budget()
            return result
        floor, runs = native_repeat_floor(infer_bounded, model,
                                           repeats, list(packed), answer_count)
        tolerance = export_tolerance(floor)
        check_budget()
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(destination, safe_serialization=True)
        from safetensors.torch import load_file
        assert_adapter_equal(saved, load_file(str(destination / 'adapter_model.safetensors')))
        model.config.save_pretrained(destination)
        if fixture_only:
            processor.save_pretrained(destination)
        else:
            for name in processor_files:
                shutil.copyfile(processor_source / name, destination / name)
            copied_digest, copied_files = processor_assets_digest(destination)
            if copied_digest != processor_digest or copied_files != processor_files:
                raise ExportError('copied processor assets changed')
        check_budget()
        relocated_devices = _relocate_source(model, source_relocator, exclusive_gpu=exclusive_gpu)
        if exclusive_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        lifecycle = dict(reference_devices=reference_devices, source_devices_before_children=relocated_devices,
                         relocation_callback=source_relocator is not None, exclusive_gpu=exclusive_gpu,
                         source_relocation_persistent=True, optimizer_state_release='caller_responsibility')
        results = []
        with tempfile.TemporaryDirectory() as temp:
            request = Path(temp) / 'request.pt'
            torch.save(dict(base=str(base_dir), adapter=str(destination), fixtures=packed, device=child_device, dtype=base_dtype,
                            source_dtype=source_dtype), request)
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='1')
            if child_device == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            for i in range(2):
                output = Path(temp) / f'result{i}.pt'
                check_budget()
                child_timeout = timeout if deadline is None else min(timeout, deadline - time.monotonic())
                try:
                    proc = subprocess.run([sys.executable, '-I', '-c', _CHILD, str(request), str(output)],
                                          env=env, capture_output=True, text=True, timeout=child_timeout)
                except subprocess.TimeoutExpired as exc:
                    from .pipeline import BudgetExpired
                    raise BudgetExpired('native export child timeout; incomplete') from exc
                check_budget()
                if proc.returncode:
                    raise ExportError(f'fresh native reload failed: {proc.stderr[-4000:]}')
                result = torch.load(output, weights_only=True)
                tensor_check = assert_adapter_equal(saved, result['adapter'])
                comparison = compare_to_reference(runs[0], result['outputs'], tolerance, list(packed), answer_count)
                if not comparison['passed'] or result['project_imports']:
                    raise ExportError('fresh native prediction/import proof failed')
                results.append(dict(comparison, adapter=tensor_check, pid=result['pid'],
                                    project_imports=result['project_imports'], delta_backend=result['delta_backend']))
        if len({r['pid'] for r in results} | {os.getpid()}) != 3:
            raise ExportError('reload processes are not independent')
        check_budget()
        report = dict(passed=True, identity=dict(identity), fixtures=list(packed),
                      native_repeat_floor=floor, tolerance=tolerance, reloads=results, lifecycle=lifecycle,
                      processor_assets=dict(fixture_only=fixture_only, sha256=processor_digest))
        (destination / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    finally:
        for module, training in training_modes:
            module.training = training
