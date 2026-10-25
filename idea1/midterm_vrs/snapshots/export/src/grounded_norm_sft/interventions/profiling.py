"""Pre-Seal-B complete-update measurements. No selection, exports or authority."""
import gc
from pathlib import Path
import time

from ..hashing import sha256_file
from ..io_utils import read_json, verify_sidecar, write_once_json
from .cache import digest

UNMEASURED = ('cache_seconds', 'native_dev_seconds', 'c2_seconds', 'io_seconds')


def validate_profile_launch(launch):
    """This phase uses its own explicit authority, never hybrid_train/Seal B."""
    from .potency import aggregate
    profile = launch['profile']
    if 'native_acceptance' in profile:
        from .native_acceptance import validate_spec
        validate_spec(profile['native_acceptance'])
    if type(profile.get('measure_overheads', False)) is not bool:
        raise ValueError('measure_overheads must be boolean')
    arms = profile['selected_core_arms']
    if (not isinstance(arms, list) or not arms or len(set(arms)) != len(arms) or
            any(a not in ('H-T', 'H-R', 'F-T', 'D-T') for a in arms)):
        raise ValueError('explicit unique selected_core_arms required')
    if (type(launch['seed']) is not int or launch['seed'] not in (17, 29, 43) or
            launch['horizon']['epochs'] != 3 or launch['split'] != 'train'):
        raise ValueError('train-only profile requires frozen seed17/29/43 and common three-epoch B1 horizon')
    if type(profile['complete_updates_per_arm']) is not int or profile['complete_updates_per_arm'] < 1:
        raise ValueError('positive integer complete_updates_per_arm required')
    for key in ('train_cache_index', 'dev_cache_index', 'potency'):
        verify_sidecar(launch[key])
        if sha256_file(launch[key]) != launch[key + '_sha256']:
            raise ValueError('profile evidence bytes mismatch: ' + key)
    potency = read_json(launch['potency'], verify=True)
    from .eta050 import launch_protocol
    dose_protocol = launch_protocol(launch)
    if dose_protocol is not None:
        if any(arm not in ('H-T','H-R') for arm in arms):
            raise ValueError('eta050 amendment authorizes H-T/H-R only')
        if (launch['potency_sha256'] != dose_protocol.as_dict()['diagnostic_sha256'] or
            potency.get('cache_index_sha256') != launch['dev_cache_index_sha256']):
            raise ValueError('eta050 profile must bind actual diagnostic and dev cache')
        from .training_cache_source import verify_cache_sources
        verify_cache_sources(launch)
        return potency
    if (aggregate(potency)['status'] != 'PASS' or potency.get('evidence_kind') != 'supplied_frozen_base' or
            potency.get('fixture_mode') or not potency.get('donor_sources_sha256') or
            potency.get('cache_index_sha256') != launch['dev_cache_index_sha256']):
        raise ValueError('actual frozen-base both-family no-label potency PASS bound to dev cache required')
    return potency


def measure_updates(model, trainer, *, requested, budget):
    """Time actual backward+optimizer+scheduler windows, including the cold first one.

    Caller owns a fresh disposable model. A partial update never counts as complete.
    No checkpoint or model tensor is written.
    """
    import torch
    from ..runtime.model import set_train_mode
    from .pipeline import BudgetExpired
    cuda = next(model.parameters()).device.type == 'cuda'
    sync = torch.cuda.synchronize if cuda else lambda: None
    if cuda:
        sync()
        torch.cuda.reset_peak_memory_stats()
    set_train_mode(model)
    rows, reason = [], None
    guard = model.register_forward_pre_hook(lambda *unused: budget.check())
    entered = time.monotonic()
    try:
        for update in range(min(requested, trainer.total_updates)):
            budget.check()
            sync()
            started = time.monotonic()
            result = trainer.run_update(update)
            sync()
            elapsed = time.monotonic() - started
            window = trainer.updates[update]
            rows.append(dict(update=update, seconds=elapsed, window_size=result['window_size'],
                ordinals=[r['ordinal'] for r in window], families=[r['site_family'] for r in window],
                layers=[r['site_layer'] for r in window], duplicate_skips=result['duplicate_skips']))
    except BudgetExpired as exc:
        reason = str(exc)
    finally:
        guard.remove()
    if len(rows) < requested and reason is None:
        reason = 'frozen schedule exhausted before requested update count'
    seconds = [row['seconds'] for row in rows]
    return dict(status='COMPLETE' if len(rows) >= requested else 'INCOMPLETE', reason=reason,
        requested_complete_updates=requested, completed_updates=len(rows), updates=rows,
        wall_seconds=time.monotonic()-entered, cold_update_seconds=seconds[0] if seconds else None,
        warm_update_seconds=seconds[1:], max_complete_update_seconds=max(seconds) if seconds else None,
        mean_complete_update_seconds=sum(seconds)/len(seconds) if seconds else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
        peak_memory_status='MEASURED_CUDA' if cuda else 'UNAVAILABLE_CPU', device=str(next(model.parameters()).device))


def measure_overheads(model, dev, paths, cohort, *, pins, alignment_ids, seed, operator, arm, budget, checkpoint, dose_protocol=None):
    """Frozen-base input-only clean dev and prescribed C2, with no vote join."""
    import torch
    from .forward import NativeForward
    from .data import SupportInputs
    from . import manipulation
    from .training import gate_health
    from contextlib import ExitStack
    native = NativeForward(model, alignment_ids=alignment_ids)
    native.bind(dev)
    supports = SupportInputs(dev, paths, pins=pins, image_sha256=dev.image_sha256,
                             alignment_ids=alignment_ids)
    ident = manipulation.identity(checkpoint_sha256=checkpoint, seed=seed,
        operator_sha256=operator, serialized=dev, records=supports, schedule=cohort[:32], dose_protocol=dose_protocol)
    selected = {a['example_id'] for a in cohort[:32]}
    clean = {}
    cuda = next(model.parameters()).device.type == 'cuda'
    sync = torch.cuda.synchronize if cuda else lambda: None
    guard = model.register_forward_pre_hook(lambda *unused: budget.check())
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        sync(); started = time.monotonic()
        for key in dev:
            budget.check()
            with torch.no_grad(), ExitStack() as stack:
                if key in selected:
                    gate_health(stack, native, dev[key], supports[key]['payload']['support'])
                logits, ids = native('normlens', dev[key])
                if key in selected:
                    # Immediately detach; retain no full-vocabulary GPU logits.
                    single = dict(ident, assignments=[a for a in cohort[:32] if a['example_id'] == key])
                    clean[key] = manipulation.record_clean(single, [(key, logits, ids)])['rows'][0]
        sync(); native_seconds = time.monotonic()-started
        predictions = dict(identity=ident, rows=[clean[a['example_id']] for a in cohort[:32]])
        budget.check(); sync(); started = time.monotonic()
        raw = manipulation.run(model, dev, frozen_assignments=cohort, clean_predictions=predictions,
            checkpoint_sha256=checkpoint, seed=seed, operator_sha256=operator, arm=arm,
            cache_paths=paths, pins=pins, image_sha256=dev.image_sha256, alignment_ids=alignment_ids, dose_protocol=dose_protocol)
        sync(); c2_seconds = time.monotonic()-started
        return dict(status='COMPLETE', native_dev_seconds=native_seconds, c2_seconds=c2_seconds,
                    dev_count=len(dev), c2_perturbed_forwards=64, c2_floor_forwards=80,
                    evidence_sha256=digest(raw), checkpoint_kind='fresh_base_with_initial_lora',
                    no_vote_join=True, identity=ident)
    finally:
        guard.remove()
        for module, mode in modes:
            module.training = mode


def measure_checkpoint_io(trainer, completed, directory, budget):
    """Time actual checkpoint serialization+verification, then remove disposable bytes."""
    from tempfile import TemporaryDirectory
    from ..runtime.checkpoint import save_checkpoint, verify_checkpoint_dir
    budget.check()
    Path(directory).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with TemporaryDirectory(prefix='preseal-disposable-io-', dir=directory) as temporary:
        path = Path(temporary) / 'checkpoint'
        save_checkpoint(path, trainer=trainer, next_update=completed,
            epoch=trainer.epoch_of(completed-1), kind='preseal_disposable_profile',
            identity=trainer.identity, log=[], extra=dict(scientific_trained_artifact=False))
        verify_checkpoint_dir(path, temporary)
        size = sum(p.stat().st_size for p in path.iterdir())
        elapsed = time.monotonic()-started
    return dict(status='MEASURED', seconds=elapsed, bytes=size,
                scope='one completed-update checkpoint serialization and verification; deletion excluded',
                retained_artifact=False)


def execute_preseal(serializer, launch, train_inputs, *, pins, alignment_ids, runtime_pins, budget):
    """Validate actual streams before weight loading; reload identical base for every arm."""
    import torch
    from .data import ManifestInputs
    from .pipeline import validate_index, BudgetExpired
    from .trainer import HybridTrainer
    from .potency import assignments
    from ..runtime.model import load_model, delta_dispatch_inventory
    from ..runtime.lora import apply_lora
    from ..schedules import build_training_schedule
    potency = validate_profile_launch(launch)
    from .eta050 import launch_protocol
    dose_protocol = launch_protocol(launch)
    data = launch['data']
    dev = ManifestInputs(data['normlens_manifest'], data['normlens_manifest_sha256'], serializer,
        task='normlens', split='dev', pins=pins, image_dir=data['normlens_image_dir'])
    dev._identity['launch_provenance'] = train_inputs.identity['launch_provenance']
    from .training_cache_source import bind_cache_source
    bind_cache_source(launch, train_inputs, 'train')
    bind_cache_source(launch, dev, 'dev')
    train_paths, _ = validate_index(launch['train_cache_index'], train_inputs, pins=pins, alignment_ids=alignment_ids)
    dev_paths, eligible = validate_index(launch['dev_cache_index'], dev, pins=pins, alignment_ids=alignment_ids)
    if potency['pins'] != pins or potency['assignments'] != assignments(eligible):
        raise ValueError('profile potency pins/cohort differ from actual streams')
    nb = ManifestInputs(data['normbank_manifest'], data['normbank_manifest_sha256'], serializer,
        task='normbank', split='train', pins=pins)
    dispatch = delta_dispatch_inventory()
    auth = launch['authorization']
    evidence = dict(schema='gn_hybrid_preseal_update_profile_v1', phase='hybrid_profile_preseal',
        source_snapshot_sha256=auth['source_snapshot_sha256'], source_manifest_sha256=auth['source_manifest_sha256'],
        runtime_pins_sha256=train_inputs.identity['launch_provenance']['runtime_pins_sha256'],
        approval_sha256=auth['approval_sha256'], reservation_id=auth['reservation_id'],
        seal_a=launch['seal_a'], operator_plan=launch['operator_plan'], pins=pins,
        train_source=train_inputs.identity, normbank_source=nb.identity,
        train_cache_index_sha256=launch['train_cache_index_sha256'], dev_cache_index_sha256=launch['dev_cache_index_sha256'],
        potency_sha256=launch['potency_sha256'], seed=launch['seed'], epochs=3,
        selected_core_arms=launch['profile']['selected_core_arms'], arms={}, dispatch=dispatch,
        overheads={key: dict(status='UNAVAILABLE', reason='not measured by complete-update-only profile') for key in UNMEASURED},
        seal_b_feasibility_complete=False, blocking_fields=list(UNMEASURED), intervention_training_allowed=False,
        scientific_trained_artifact=False)
    if dose_protocol is not None:
        evidence['dose_protocol'] = dose_protocol.as_dict()
    output = Path(launch['paths']['output_root']) / ('preseal-profile-' + auth['reservation_id'] + '.json')
    if 'native_acceptance' in launch['profile']:
        from .native_acceptance import validate_spec, run_checks
        spec = launch['profile']['native_acceptance']
        validate_spec(spec)
        ids = spec['fixture_ids']
        bank = spec['normbank_fixture_id']
        if any(key not in train_inputs for key in ids) or bank not in nb:
            raise ValueError('native acceptance fixtures must belong to actual TRAIN streams')
        fixed = {key: train_inputs[key] for key in ids}
        fixed_nb = {bank: nb[bank]}
        if any(len(item.input_ids) > spec['max_prompt_tokens'] for item in [*fixed.values(), *fixed_nb.values()]):
            raise ValueError('fixed native acceptance fixture exceeds cap; no adaptive replacement/truncation')
        model = trainer = None
        try:
            budget.check()
            model, receipt = load_model(launch['paths']['model_dir'], runtime_pins)
            model, _ = apply_lora(model, launch['seed'])
            schedule = build_training_schedule(ids, [bank], launch['seed'], 3, 'H-T')
            trainer = HybridTrainer(model, schedule, fixed,
                {key: train_inputs.targets[key] for key in ids}, fixed_nb, {bank: nb.targets[bank]},
                cache_paths={key:train_paths[key] for key in ids}, pins=pins,
                image_sha256={key:train_inputs.image_sha256[key] for key in ids},
                alignment_ids=alignment_ids, protocol_identity=dict(preseal_native_acceptance=True,
                    train_source=train_inputs.identity, normbank_source=nb.identity,
                    fixture_spec=spec, operator_plan_sha256=launch['operator_plan']['sha256']),
                eta=.5 if dose_protocol else .25, dose_protocol=dose_protocol)
            identity = dict(source_snapshot_sha256=auth['source_snapshot_sha256'],
                source_manifest_sha256=auth['source_manifest_sha256'],
                runtime_pins_sha256=evidence['runtime_pins_sha256'],
                train_cache_index_sha256=launch['train_cache_index_sha256'],
                model_load_receipt=receipt, fixture_spec=spec,
                operator_plan_sha256=launch['operator_plan']['sha256'])
            if dose_protocol is not None:
                identity['dose_protocol'] = dose_protocol.as_dict()
            evidence['native_acceptance'] = run_checks(trainer,
                output=Path(launch['paths']['output_root'])/('native-acceptance-'+auth['reservation_id']),
                identity=identity,budget=budget)
        finally:
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    initial = None
    for arm in evidence['selected_core_arms']:
        model = trainer = result = None
        try:
            budget.check()
            model, receipt = load_model(launch['paths']['model_dir'], runtime_pins)
            budget.check()
            model, _ = apply_lora(model, launch['seed'])
            schedule = build_training_schedule(list(train_inputs), list(nb), launch['seed'], 3, arm)
            trainer = HybridTrainer(model, schedule, train_inputs, train_inputs.targets, nb, nb.targets,
                cache_paths=train_paths, pins=pins, image_sha256=train_inputs.image_sha256,
                alignment_ids=alignment_ids, protocol_identity=dict(preseal_profile=True,
                    operator_plan_sha256=launch['operator_plan']['sha256']),
                eta=.5 if dose_protocol else .25, dose_protocol=dose_protocol)
            identity = dict(frozen_model=trainer.identity['frozen_model'], adapter_init=trainer.identity['adapter_init'])
            if initial is not None and initial != identity:
                raise ValueError('profile arms must start from identical fresh base and LoRA initialization')
            initial = identity
            overhead = None
            if launch['profile'].get('measure_overheads', False):
                overhead = measure_overheads(model, dev, dev_paths, assignments(eligible), pins=pins,
                    alignment_ids=alignment_ids, seed=launch['seed'], operator=launch['operator_plan']['sha256'],
                    arm=arm, budget=budget, checkpoint=trainer.identity['adapter_init'], dose_protocol=dose_protocol)
            result = measure_updates(model, trainer, requested=launch['profile']['complete_updates_per_arm'], budget=budget)
            result.update(overhead_measurements=overhead, arm=arm, seed=launch['seed'], schedule_sha256=digest(schedule),
                schedule_total_updates=trainer.total_updates, initialization=identity, model_load_receipt=receipt)
            if overhead is not None and result['completed_updates']:
                overhead['checkpoint_io'] = measure_checkpoint_io(trainer, result['completed_updates'],
                    launch['paths']['output_root'], budget)
            evidence['arms'][arm] = result
        except BudgetExpired as exc:
            evidence['arms'][arm] = dict(result or {}, status='INCOMPLETE', reason=str(exc))
            evidence['arms'][arm].setdefault('completed_updates', 0)
            evidence['arms'][arm].setdefault('requested_complete_updates', launch['profile']['complete_updates_per_arm'])
        finally:
            # Dispose optimizer, graph/model and CUDA allocations before the next base load.
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if all(a.get('overhead_measurements') for a in evidence['arms'].values()):
        evidence['blocking_fields'] = ['cache_seconds', 'io_seconds']
        for field in ('native_dev_seconds', 'c2_seconds'):
            evidence['overheads'][field] = dict(status='MEASURED_PER_ARM', location='arms.*.overhead_measurements')
    evidence['status'] = 'COMPLETE' if all(a['status'] == 'COMPLETE' for a in evidence['arms'].values()) else 'INCOMPLETE'
    write_once_json(output, evidence)
    return dict(path=str(output), status=evidence['status'], seal_b_feasibility_complete=False,
                blocking_fields=evidence['blocking_fields'], intervention_training_allowed=False)


def run_fixture(output):
    """Disposable random CPU native gradients; no authority or scientific model."""
    from .fixture import training_fixture
    from .pipeline import Budget
    measurements = []
    for index, arm in enumerate(('H-T', 'H-R')):
        model, trainer, _, _ = training_fixture(Path(output) / ('setup%d' % index), epochs=3, arm=arm)
        measured = measure_updates(model, trainer, requested=1, budget=Budget(2, .1))
        measured['arm'] = arm
        measured['initialization'] = {k: trainer.identity[k] for k in ('frozen_model', 'adapter_init')}
        measurements.append(measured)
        del trainer, model
        gc.collect()
    fresh = measurements[0]['initialization'] == measurements[1]['initialization']
    if not fresh:
        raise ValueError('fresh fixture initialization differs')
    evidence = dict(fixture_mode=True, measurements=measurements, fresh_initialization_verified=fresh,
                    intervention_training_allowed=False, scientific_trained_artifact=False)
    write_once_json(Path(output) / 'tiny-preseal-profile.json', evidence)
    return evidence
