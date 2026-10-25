"""Selected completed hybrid checkpoint -> ordinary PEFT export, without optimizer restore."""
from pathlib import Path
from ..hashing import sha256_file
from ..io_utils import read_json
from ..runtime.checkpoint import verify_checkpoint_dir
from ..runtime.lora import adapter_sha256, load_adapter_tensors
from ..runtime.tensorio import load_tensors
from ..runtime.train import select_epoch
from .cache import digest


def selected_adapter(model, trainer, root, budget):
    """Recheck completion and native epoch selection; load only the chosen FP32 adapter."""
    budget.check()
    root = Path(root)
    identity = read_json(root / 'identity.json', verify=True)
    selection = read_json(root / 'selection.json', verify=True)
    if identity != trainer.identity or selection['identity'] != identity:
        raise ValueError('selected hybrid run identity differs from reconstructed run')
    if read_json(root / 'schedule.json', verify=True) != trainer.schedule:
        raise ValueError('selected hybrid schedule changed')
    results = [read_json(root / 'dev_results' / ('epoch%d.json' % e), verify=True) for e in range(trainer.epochs)]
    expected = select_epoch(results, trainer.epochs)
    chosen = next(r for r in results if r['epoch'] == expected['selected_epoch'])
    expected['checkpoint'] = chosen['checkpoint']
    if selection != dict(expected, identity=identity):
        raise ValueError('selected checkpoint differs from complete native dev selection')
    # A selection file alone is insufficient: prove the final epoch really completed.
    for record in results:
        budget.check()
        ck = record['checkpoint']
        verify_checkpoint_dir(ck['dir'], root)
        if sha256_file(Path(ck['dir']) / 'MANIFEST.json') != ck['manifest_sha256']:
            raise ValueError('selected epoch checkpoint manifest changed')
        state = read_json(Path(ck['dir']) / 'state.json', verify=True)
        if (state['identity'] != identity or state['kind'] != 'epoch' or state['epoch'] != record['epoch'] or
                state['next_update'] != ck['next_update'] or state['total_updates'] != trainer.total_updates):
            raise ValueError('epoch checkpoint identity/cursor differs')
        if record['epoch'] == trainer.epochs - 1 and state['next_update'] != trainer.total_updates:
            raise ValueError('hybrid horizon is incomplete')
    ck = selection['checkpoint']
    state = read_json(Path(ck['dir']) / 'state.json', verify=True)
    adapter, _ = load_tensors(Path(ck['dir']) / 'adapter.safetensors')
    if adapter_sha256(adapter) != state['adapter_sha256']:
        raise ValueError('selected adapter bytes differ')
    load_adapter_tensors(model, adapter)
    budget.check()
    # Export never restores AdamW or RNG, and releases any caller-created state.
    trainer.optimizer.state.clear()
    trainer.optimizer.zero_grad(set_to_none=True)
    return dict(selection_sha256=sha256_file(root / 'selection.json'),
                checkpoint_sha256=ck['manifest_sha256'], adapter_sha256=state['adapter_sha256'],
                run_identity_sha256=digest(identity), schedule_sha256=identity['schedule_sha256'],
                support_cache_sha256=identity['support_cache_sha256'])


def export_selected(model, trainer, root, *, processor, fixtures, base_dir, destination,
                    identity, budget, timeout, processor_dir=None, fixture_only=False, child_device='cpu'):
    from .exporting import export_native_adapter
    bound = selected_adapter(model, trainer, root, budget)
    if set(bound) & set(identity):
        raise ValueError('caller cannot replace selected checkpoint identities')
    bound.update(identity)
    if set(identity) & {'arm','seed'}:
        raise ValueError('caller cannot replace selected arm/seed')
    bound.update(arm=trainer.arm, seed=str(trainer.schedule['run_seed']))
    budget.check()
    report = export_native_adapter(model, destination, base_dir=base_dir, processor=processor,
        fixtures=fixtures, identity=bound, processor_dir=processor_dir, fixture_only=fixture_only,
        timeout=timeout, child_device=child_device, base_dtype='float32' if fixture_only else 'bfloat16',
        source_relocator=lambda source: source.to('cpu'), deadline=budget.deadline-budget.reserve_seconds)
    budget.check()
    return dict(status='complete', fixture_mode=fixture_only, production_authorized=False, production_mode=not fixture_only,
                export_dir=str(destination), report=report)


def execute_loaded(model, serializer, launch, train_inputs, *, pins, alignment_ids, budget):
    """Production dispatch after explicit hybrid_export and evidence-bound Seal B gates."""
    from .data import ManifestInputs
    from .pipeline import validate_index
    from .trainer import HybridTrainer
    from .gates import validate_seal_b
    from ..runtime.pins import load_pins
    from ..runtime.lora import apply_lora
    from ..schedules import build_training_schedule
    from .training import REDUCTION
    seal, _ = validate_seal_b(launch, load_pins()[1])
    from .eta050 import launch_protocol
    dose_protocol = launch_protocol(launch)
    d = launch['data']
    nb = ManifestInputs(d['normbank_manifest'], d['normbank_manifest_sha256'], serializer,
                        task='normbank', split='train', pins=pins)
    budget.check()
    from .training_cache_source import bind_cache_source
    bind_cache_source(launch, train_inputs, 'train')
    paths, _ = validate_index(launch['train_cache_index'], train_inputs, pins=pins, alignment_ids=alignment_ids)
    schedule = build_training_schedule(list(train_inputs), list(nb), launch['seed'], 3, launch['arm'])
    model, _ = apply_lora(model, launch['seed'])
    protocol = dict(seal_b_sha256=launch['seal_b']['sha256'], checkpoint_reduction=REDUCTION,
        operator_plan_sha256=launch['operator_plan']['sha256'], chronology=seal['chronology'])
    trainer = HybridTrainer(model, schedule, train_inputs, train_inputs.targets, nb, nb.targets,
        cache_paths=paths, pins=pins, image_sha256=train_inputs.image_sha256,
        alignment_ids=alignment_ids, protocol_identity=protocol,
        eta=.5 if dose_protocol else .25, dose_protocol=dose_protocol)
    root = Path(launch['paths']['training_run_dir'])
    if root.name != 'hybrid_' + launch['arm'] + '_' + digest(trainer.identity)[:16]:
        raise ValueError('explicit training_run_dir differs from reconstructed hybrid identity')
    dev = ManifestInputs(d['normlens_manifest'], d['normlens_manifest_sha256'], serializer,
        task='normlens', split='dev', pins=pins, image_dir=d['normlens_image_dir'])
    dev._identity['launch_provenance'] = train_inputs.identity['launch_provenance']
    bind_cache_source(launch, dev, 'dev')
    validate_index(launch['dev_cache_index'], dev, pins=pins, alignment_ids=alignment_ids)
    budget.check()
    fixtures = {key: dev[key] for key in launch['export']['fixture_ids']}
    return export_selected(model, trainer, root, processor=serializer.processor, fixtures=fixtures,
        base_dir=launch['paths']['model_dir'], processor_dir=launch['paths']['processor_dir'],
        destination=Path(launch['paths']['output_root']) / 'native-adapter', budget=budget,
        timeout=launch['export']['child_timeout_seconds'], child_device='cuda:0',
        identity=dict(protocol=launch['operator_plan']['sha256'], model=pins['model_sha256'],
            processor=pins['processor_sha256'], template=pins['template_sha256'],
            seal_b=launch['seal_b']['sha256'], source_snapshot=launch['authorization']['source_snapshot_sha256'],
            source_manifest=launch['authorization']['source_manifest_sha256'],
            **({'dose_protocol_sha256':digest(dose_protocol.as_dict())} if dose_protocol else {}),
            **({'export_consumer_'+key:value for key,value in launch['export_consumer'].items()}
               if launch.get('export_consumer') else {})))
