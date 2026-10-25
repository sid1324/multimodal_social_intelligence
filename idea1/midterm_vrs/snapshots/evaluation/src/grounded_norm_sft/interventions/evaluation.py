"""Ordinary native evaluation of eligible hybrid PEFT exports; no interventions."""
from pathlib import Path
import math
import time

from ..hashing import sha256_file
from ..io_utils import read_json
from ..runtime.errors import RuntimeContractError
from .cache import digest

ARMS = ('H-T', 'H-R', 'F-T', 'D-T')
SCORING_SEAL_SHA256 = '61f71b05ba9b46193dde8a7ab1b48e76c1c53e3f5365004652882980d754f1c6'


def validate_scoring_seal(launch):
    seal=launch['scoring_seal']
    if seal['sha256']!=SCORING_SEAL_SHA256 or sha256_file(seal['path'])!=SCORING_SEAL_SHA256:
        raise ValueError('fixed baseline scoring seal bytes mismatch')
    return SCORING_SEAL_SHA256


def verify_export(launch, pins):
    """Validate the hybrid export contract (which has no baseline export schema)."""
    e = launch['eval']
    root = Path(e['adapter_dir'])
    if sha256_file(root / 'export_report.json') != e['export_report_sha256']:
        raise ValueError('hybrid export report bytes changed')
    if sha256_file(root / 'adapter_config.json') != e['adapter_config_sha256']:
        raise ValueError('hybrid adapter configuration bytes changed')
    from ..constants import LORA_RANK, LORA_ALPHA, LORA_DROPOUT
    from ..runtime.lora import target_regex
    config = read_json(root / 'adapter_config.json')
    required = dict(peft_type='LORA', r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                    bias='none', target_modules=target_regex())
    if any(config.get(k) != v for k,v in required.items()) or any(config.get(k) for k in
            ('rank_pattern','alpha_pattern','use_dora','use_rslora','modules_to_save','lora_bias')):
        raise ValueError('hybrid adapter configuration differs from frozen LoRA contract')
    report = read_json(root / 'export_report.json')  # hybrid exporter writes no sidecar
    identity = report['identity']
    expected = dict(arm=launch['arm'], seed=str(launch['seed']),
        protocol=launch['operator_plan']['sha256'], seal_b=launch['seal_b']['sha256'],
        model=pins['model_sha256'], processor=pins['processor_sha256'], template=pins['template_sha256'],
        source_snapshot=launch['authorization']['source_snapshot_sha256'],
        source_manifest=launch['authorization']['source_manifest_sha256'])
    if launch['arm'] not in ARMS or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('hybrid export arm/seed/protocol/assets mismatch')
    exporter=e['export_consumer']
    if set(exporter)!={'code_root','source_snapshot_sha256','source_manifest_sha256'} or any(
            identity.get('export_consumer_'+k)!=v for k,v in exporter.items()):
        raise ValueError('actual export consumer differs from report binding')
    from .eta050 import launch_protocol
    protocol=launch_protocol(launch)
    if protocol is not None and identity.get('dose_protocol_sha256')!=digest(protocol.as_dict()):
        raise ValueError('export dose protocol differs from validated training amendment')
    if protocol is None and 'dose_protocol_sha256' in identity:
        raise ValueError('amended export requires explicit training amendment')
    validate_scoring_seal(launch)
    if report.get('passed') is not True or report['processor_assets'] != dict(fixture_only=False, sha256=pins['processor_sha256']):
        raise ValueError('production native export PASS and original processor assets required')
    from ..runtime.export import export_tolerance
    if report['tolerance'] != export_tolerance(report['native_repeat_floor']) or not report['fixtures']:
        raise ValueError('native repeat floor/tolerance/fixed fixtures invalid')
    reloads = report['reloads']
    if (len(reloads) != 2 or len({r['pid'] for r in reloads}) != 2 or
            any(r.get('passed') is not True or r.get('project_imports') != [] or
                r.get('delta_backend') != 'torch_native_fallback' or
                r.get('adapter', {}).get('bitwise_equal') is not True or
                not math.isfinite(r['max_abs_logprob_diff']) or r['max_abs_logprob_diff'] > report['tolerance'] for r in reloads)):
        raise ValueError('two independent clean native export reloads required')
    if report['lifecycle'].get('exclusive_gpu') is not True or report['lifecycle']['source_devices_before_children'] != ['cpu']:
        raise ValueError('exclusive native export lifecycle proof required')
    from ..runtime.pins import processor_assets_digest
    if processor_assets_digest(root)[0] != pins['processor_sha256']:
        raise ValueError('exported processor assets changed')
    from ..runtime.lora import adapter_sha256
    from ..runtime.tensorio import load_tensors
    adapter, _ = load_tensors(root / 'adapter_model.safetensors')
    selection = read_json(Path(launch['paths']['training_run_dir']) / 'selection.json', verify=True)
    if (sha256_file(Path(launch['paths']['training_run_dir']) / 'selection.json') != identity['selection_sha256'] or
            selection['checkpoint']['manifest_sha256'] != identity['checkpoint_sha256'] or
            digest(selection['identity']) != identity['run_identity_sha256'] or
            selection['identity']['arm'] != launch['arm'] or
            selection['identity']['protocol']['seal_b_sha256'] != launch['seal_b']['sha256'] or
            selection['identity']['schedule_sha256'] != identity['schedule_sha256'] or
            selection['identity']['support_cache_sha256'] != identity['support_cache_sha256']):
        raise ValueError('export is not the bound selected hybrid checkpoint')
    if protocol is not None and (selection['identity']['eta'] != .5 or
            selection['identity']['protocol'].get('dose_protocol') != protocol.as_dict()):
        raise ValueError('selected hybrid checkpoint has a different training dose')
    from ..runtime.checkpoint import verify_checkpoint_dir
    from ..runtime.export import assert_adapter_equal
    checkpoint = Path(selection['checkpoint']['dir'])
    verify_checkpoint_dir(checkpoint, launch['paths']['training_run_dir'])
    if sha256_file(checkpoint / 'MANIFEST.json') != identity['checkpoint_sha256']:
        raise ValueError('selected checkpoint manifest changed')
    selected, _ = load_tensors(checkpoint / 'adapter.safetensors')
    if adapter_sha256(selected) != identity['adapter_sha256']:
        raise ValueError('selected checkpoint adapter digest changed')
    # Training checkpoints name the active PEFT adapter; standard PEFT export strips it.
    selected = {k.replace('.default.', '.'):v for k,v in selected.items()}
    assert_adapter_equal(selected, adapter)
    return report


def run_records(out, identity, records, logits_fn, letters, *, budget=None, reserve_seconds=0):
    """Shared full-vocabulary metrics and immutable per-ID restart semantics."""
    from ..runtime.evaluate import ResumableNativeEval
    run = ResumableNativeEval(out, identity, records, logits_fn, letters,
        seed=identity['seed'], condition=identity['condition'], task=identity['task'], arm=identity['arm'],
        budget=budget, first_forward_reserve_s=reserve_seconds)
    run.open()
    result = run.run()
    if result['status'] == 'paused':
        return dict(result, out=str(out))
    manifest = run.finalize(dict(native=True, interventions=False))
    return dict(status='complete' if manifest['complete'] else 'incomplete', out=str(out),
                predicted=manifest['predicted'], failed=manifest['failed'])


def load_verified_adapter(base, launch, report):
    """Preserve saved FP32 LoRA and prove actual loaded identity before any forward."""
    from collections import Counter
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from ..runtime.errors import ExportError
    from ..runtime.export import assert_adapter_equal
    from ..runtime.lora import adapter_sha256
    from ..runtime.tensorio import load_tensors
    e = launch['eval']
    root = Path(e['adapter_dir'])
    if (sha256_file(root/'export_report.json') != e['export_report_sha256'] or
            sha256_file(root/'adapter_config.json') != e['adapter_config_sha256']):
        raise ExportError('adapter export/config changed before actual load')
    before = dict(Counter(str(p.dtype) for p in base.parameters()))
    # PEFT promotes adapter slots before copying the FP32 state. Promoting after
    # a BF16 load would preserve already rounded values and fail the exact guard.
    model = PeftModel.from_pretrained(base, e['adapter_dir'], local_files_only=True,
                                     is_trainable=False, autocast_adapter_dtype=True)
    saved, _ = load_tensors(root/'adapter_model.safetensors')
    loaded = {k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()}
    if any(not torch.isfinite(t).all() for tensors in (saved, loaded) for t in tensors.values()):
        raise ExportError('saved/loaded adapter contains nonfinite tensors')
    equality = assert_adapter_equal(saved, loaded)
    named = {n:p.detach().cpu() for n,p in model.named_parameters() if 'lora_' in n}
    checkpoint_adapter_sha = adapter_sha256(named)
    if len(named) != equality['tensors'] or checkpoint_adapter_sha != report['identity']['adapter_sha256']:
        raise ExportError('actual loaded adapter differs from selected checkpoint identity')
    after = dict(Counter(str(p.dtype) for n,p in model.named_parameters() if 'lora_' not in n))
    if before != after:
        raise ExportError('adapter load changed base parameter dtype counts')
    proof = dict(schema='gn_hybrid_loaded_adapter_v1', checked_before_inference=True,
        loader_autocast_adapter_dtype=True, tensor_count=equality['tensors'],
        saved_dtype='torch.float32', loaded_dtype='torch.float32', finite=True, bitwise_equal=True,
        saved_tensor_sha256=adapter_sha256(saved), loaded_tensor_sha256=adapter_sha256(loaded),
        checkpoint_adapter_sha256=checkpoint_adapter_sha,
        adapter_file_sha256=sha256_file(root/'adapter_model.safetensors'),
        export_report_sha256=e['export_report_sha256'], adapter_config_sha256=e['adapter_config_sha256'],
        checkpoint_sha256=report['identity']['checkpoint_sha256'],
        base_parameter_dtype_counts_before=before, base_parameter_dtype_counts_after=after)
    return model, proof


def execute_loaded(model, serializer, launch, *, pins, budget):
    from ..runtime.data_io import ImageLoader
    from ..runtime.export import assert_native_source
    from ..runtime.model import NativeForward, TaskForward
    e, d = launch['eval'], launch['data']
    report = verify_export(launch, pins)
    budget.check()
    model, loaded_adapter_verification = load_verified_adapter(model, launch, report)
    inventory = assert_native_source(model)
    if inventory['hooks']:
        raise RuntimeContractError('hybrid evaluation forbids hooks')
    model.eval()
    records,task,condition,letters,mapping=evaluation_cohort(launch)
    images=ImageLoader(d['normlens_image_dir']) if e['task']=='normlens' else None
    identity = dict(schema='gn_hybrid_eval_v1', phase='hybrid_eval', arm=launch['arm'], seed=launch['seed'],
        task=e['task'], split='test', condition=condition, export_report_sha256=e['export_report_sha256'],
        loaded_adapter_verification=loaded_adapter_verification,
        checkpoint_sha256=report['identity']['checkpoint_sha256'], adapter_config_sha256=e['adapter_config_sha256'], seal_a_sha256=launch['seal_a']['sha256'],
        seal_b_sha256=launch['seal_b']['sha256'], operator_plan_sha256=launch['operator_plan']['sha256'],
        model_pins=pins, scoring_seal_sha256=validate_scoring_seal(launch),
        trained_source_snapshot_sha256=launch['authorization']['source_snapshot_sha256'],
        export_consumer=e['export_consumer'],eval_consumer=launch['eval_consumer'],
        data_manifest_sha256=d['normlens_manifest_sha256' if e['task']=='normlens' else 'normbank_manifest_sha256'],
        expected_ids_sha256=digest([r['example_id'] for r in records]), answer_letters=list(letters),
        shuffle_map_sha256=e.get('shuffle_map_sha256') if condition=='shuffle' else None)
    eos = model.config.text_config.eos_token_id
    excluded = set(eos if isinstance(eos,list) else [eos]) | {serializer.processor.tokenizer.eos_token_id, serializer.processor.tokenizer.pad_token_id}
    fwd = TaskForward(serializer, NativeForward(model, 'cuda'), images, grad=False, excluded_ids=excluded, shuffle_map=mapping)
    reserve = e['first_forward_reserve_minutes'] * 60
    if type(reserve) not in (int,float) or not math.isfinite(reserve) or reserve <= 0:
        raise ValueError('finite positive first-forward reserve required')
    class EvalBudget:
        def allows(self, seconds):
            return time.monotonic() + max(seconds, budget.reserve_seconds) < budget.deadline
    out = Path(launch['paths']['output_root']) / ('eval_' + launch['arm'] + '_' + digest(identity)[:16])
    return run_records(out, identity, records, fwd.with_meta(task), letters, budget=EvalBudget(), reserve_seconds=reserve)


def evaluation_cohort(launch):
    """Frozen held-out IDs/conditions, derived before any model forward."""
    from ..constants import NORMLENS_LETTERS, NORMBANK_LETTERS
    from ..data.normbank import retention_subset
    from ..data.eval_manifests import shuffle_maps
    from ..runtime.data_io import load_normlens,load_normbank,normlens_input,normbank_input,split_rows
    e,d=launch['eval'],launch['data']
    if e['split']!='test' or e['task'] not in ('normlens','normbank_retention'):
        raise ValueError('fixed held-out evaluation task/split required')
    if e['task']=='normlens' and e['condition'] not in ('image','text_only','shuffle'):
        raise ValueError('fixed NormLens native condition required')
    mapping = None
    if e['task'] == 'normlens':
        source = load_normlens(d['normlens_manifest'], d['normlens_manifest_sha256'])
        records = [normlens_input(r) for r in split_rows(source, 'test')]
        if len(records) != 291:
            raise ValueError('sealed NormLens test cohort must contain 291 examples')
        condition, letters = e['condition'], NORMLENS_LETTERS
        task = {'image':'normlens', 'text_only':'normlens_text_only', 'shuffle':'normlens_shuffle'}[condition]
        if condition == 'shuffle':
            mapping = read_json(e['shuffle_map'], verify=True)
            if sha256_file(e['shuffle_map']) != e['shuffle_map_sha256'] or mapping != shuffle_maps(source)['test']['mapping']:
                raise ValueError('frozen test shuffle map mismatch')
    else:
        if e['condition']!='normbank_retention':
            raise ValueError('fixed NormBank retention condition required')
        source = load_normbank(d['normbank_manifest'], d['normbank_manifest_sha256'])
        by_id = {r['example_id']:r for r in source}
        records = [normbank_input(by_id[i]) for i in retention_subset(source)]
        if len(records) != 2000:
            raise ValueError('sealed NormBank retention cohort must contain 2000 examples')
        condition, letters, task = 'normbank_retention', NORMBANK_LETTERS, 'normbank'
    return records,task,condition,letters,mapping
