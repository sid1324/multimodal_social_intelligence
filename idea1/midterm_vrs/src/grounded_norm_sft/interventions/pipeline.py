"""Operational streamed cache and bounded dev probe. No baseline runtime setup."""
import json
import os
from pathlib import Path
import time
from ..hashing import sha256_file, sha256_bytes
from ..io_utils import write_json, write_once_json, verify_sidecar
from .cache import digest, read_record, provenance
from .acquisition_source import bind_acquisition_source


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, minutes, margin, reserve_seconds=0):
        self.reserve_seconds = reserve_seconds
        self.deadline = time.monotonic() + 60 * (minutes - margin)

    def check(self):
        if time.monotonic() + self.reserve_seconds >= self.deadline:
            raise BudgetExpired('bounded walltime exhausted; resume requires a new reservation')


def cache_inputs(model, serializer, inputs, directory, *, pins, alignment_ids, image_for, budget):
    """Resume through native create-only records; final index only after exact coverage."""
    from .reference import ReferencePipeline
    root = Path(directory)
    identity = dict(source=inputs.identity, pins=pins, alignment_ids=alignment_ids)
    index_path = root / ('index-' + digest(identity) + '.json')
    progress_path = root / ('progress-' + digest(identity) + '.json')
    pipeline = ReferencePipeline(model, serializer, root / 'records', pins=pins, alignment_ids=alignment_ids)
    entries = {}
    state = dict(identity=identity, complete=False, entries=entries)
    try:
        for key in inputs:
            budget.check()
            item = inputs[key]
            image, image_id = image_for(key)
            try:
                result = pipeline.run(item, image_id=image_id, image=image, image_sha256=inputs.image_sha256[key])
            finally:
                if hasattr(image, 'close'):
                    image.close()
            entries[key] = {kind: dict(path=str(result[kind]['path'].resolve()),
                sha256=sha256_file(result[kind]['path']), record_sha256=result[kind]['record']['sha256'],
                diagnostics_path=str(result[kind]['diagnostics_path'].resolve()),
                diagnostics_sha256=sha256_file(result[kind]['diagnostics_path']))
                for kind in ('action', 'generic')}
            state['coverage'] = result['coverage']
            write_json(progress_path, state)
        state['complete'] = True
        if index_path.exists():
            verify_sidecar(index_path)
            if json.loads(index_path.read_text()) != state:
                raise ValueError('complete cache index conflict')
        else:
            write_once_json(index_path, state)
        return index_path, state
    except BudgetExpired:
        state['complete'] = False
        write_json(progress_path, state)
        raise


def validate_index(path, inputs, *, pins, alignment_ids):
    verify_sidecar(path)
    index = json.loads(Path(path).read_text())
    expected = dict(source=inputs.identity, pins=pins, alignment_ids=alignment_ids)
    if not index.get('complete') or index.get('identity') != expected or set(index['entries']) != set(inputs):
        raise ValueError('complete cache/source/policy/coverage identity mismatch')
    paths, supported = {}, []
    for key in inputs:
        for kind, entry in index['entries'][key].items():
            if (kind not in ('action', 'generic') or sha256_file(entry['path']) != entry['sha256']
                    or sha256_file(entry['diagnostics_path']) != entry['diagnostics_sha256']):
                raise ValueError('cache index bytes mismatch')
        if set(index['entries'][key]) != {'action', 'generic'}:
            raise ValueError('action/generic cache coverage missing')
        entry = index['entries'][key]['action']
        record = read_record(entry['path'], expected_provenance=provenance(inputs[key],
            **dict(pins, image_sha256=inputs.image_sha256[key]), **alignment_ids))
        if record['sha256'] != entry['record_sha256']:
            raise ValueError('cache index record mismatch')
        paths[key] = entry['path']
        if record['payload']['support']['supported']:
            supported.append(key)
    return paths, supported


def donor_source_pins(mapping, rows, image_dir):
    """Bind frozen donor files to sealed raw-file allowlists and decoded RGB IDs."""
    from ..data.images import resolve_image_path
    sources = {}
    for key, donor in mapping.items():
        matches = [r for r in rows if r['rgb_sha256'] == donor['rgb_sha256']
                   and donor['image'] in (r.get('images') or [r['image']])]
        if not matches:
            raise ValueError('frozen donor has no sealed manifest image source')
        path = resolve_image_path(image_dir, donor['image'])
        raw = sha256_file(path)
        for row in matches:
            allowed = row.get('image_file_sha256s')
            if allowed is not None and raw not in allowed:
                raise ValueError('donor raw image hash differs from sealed manifest')
        sources[key] = dict(image=donor['image'], path=str(path.resolve()),
                            rgb_sha256=donor['rgb_sha256'], raw_sha256=raw)
    return sources


def probe(model, serializer, inputs, index_path, shuffle_map, directory, *, pins, alignment_ids, image_for, budget, attempt_id, donor_sources):
    from .potency import assignments, run
    cache_paths, eligible = validate_index(index_path, inputs, pins=pins, alignment_ids=alignment_ids)
    schedule = assignments(eligible)
    # Two eager maps of at most128 inputs: NativeForward forbids duplicate task streams.
    clean, shuffled = {}, {}
    for row in schedule:
        budget.check()
        key = row['example_id']
        clean[key] = inputs[key]
        image, _ = image_for(key, shuffle_map[key])
        try:
            shuffled[key] = serializer.normlens(key, inputs._rows[key]['action'], image, text_only=False)
        finally:
            image.close()
    def guard(*unused):
        budget.check()
    hook = model.register_forward_pre_hook(guard)
    path = Path(directory) / ('potency-' + digest(dict(index_sha256=sha256_file(index_path), shuffle=shuffle_map, donor_sources=donor_sources, attempt_id=attempt_id)) + '.json')
    try:
        evidence = run(model, clean, shuffled, cache_paths={k: cache_paths[k] for k in clean},
            pins=pins, image_sha256={k: inputs.image_sha256[k] for k in clean}, alignment_ids=alignment_ids,
            shuffle_map=shuffle_map, shuffle_map_sha256=digest(shuffle_map))
        # Preserve full eligible population identity; selected cohort is unchanged by cap.
        evidence['eligible_population_sha256'] = digest(eligible)
        evidence['cache_index_sha256'] = sha256_file(index_path)
        evidence['donor_sources'] = donor_sources
        evidence['donor_sources_sha256'] = digest(donor_sources)
        evidence['intervention_training_allowed'] = False
        write_once_json(path, evidence)
        return dict(path=str(path), report=evidence['report'], intervention_training_allowed=False)
    except BudgetExpired:
        write_once_json(path.with_name(path.stem + '-incomplete.json'), dict(status='INCOMPLETE', reason='walltime exhausted',
            cache_index_sha256=sha256_file(index_path), intervention_training_allowed=False))
        raise
    finally:
        hook.remove()


def execute(launch_path, command, *, resume=False):
    from ..runtime.gates import load_launch
    from .gates import check_launch
    launch, _ = load_launch(launch_path)
    phase = launch.get('phase') if command == 'check-launch' else 'hybrid_' + command.replace('-', '_')
    entered = time.monotonic()
    report, approval, runtime_pins = check_launch(launch, phase,
        env=None if command == 'check-launch' else os.environ, audit_assets=command == 'check-launch')
    if command == 'check-launch':
        return report
    budget = Budget(launch['job']['time_limit_minutes'], launch['job']['stop_margin_minutes'])
    budget.deadline -= time.monotonic() - entered
    from .approval import mark_started
    auth, paths, data = launch['authorization'], launch['paths'], launch['data']
    mark_started(approval, auth['approval_sha256'], auth['reservation_id'], phase,
        launch['job']['time_limit_minutes'], os.environ['SLURM_JOB_ID'])
    budget.reserve_seconds = launch['job']['item_reserve_seconds']
    hook = None
    try:
        budget.check()
        from ..runtime.model import prepare_process_env, check_device, load_processor, load_model, set_determinism, receipt_chat_template_text
        prepare_process_env()
        check_device()
        set_determinism()
        processor, _ = load_processor(paths.get('processor_dir', paths['model_dir']), runtime_pins, require_pin=True)
        budget.check()
        from ..serializer import Serializer
        from .data import ManifestInputs
        serializer = Serializer(processor)
        pins = dict(model_sha256=digest(runtime_pins['model']['safetensors_shard_sha256']),
            processor_sha256=runtime_pins['model']['processor_and_chat_template_sha256'],
            template_sha256=sha256_bytes(receipt_chat_template_text(processor).encode()))
        alignment = dict(image_token_id=serializer.image_token_id, vision_start_id=serializer.vision_start_id,
            vision_end_id=serializer.vision_end_id, merge_size=serializer.merge_size)
        inputs = ManifestInputs(data['normlens_manifest'], data['normlens_manifest_sha256'], serializer,
            task='normlens', split=launch['split'], pins=pins, image_dir=data['normlens_image_dir'])
        inputs._identity['launch_provenance'] = dict(source_snapshot_sha256=auth['source_snapshot_sha256'],
            source_manifest_sha256=auth['source_manifest_sha256'], seal_a=launch['seal_a'],
            runtime_pins_sha256=report['pins_sha256'], operator_plan_sha256=launch['operator_plan']['sha256'])
        def image_for(key, donor=None):
            row = inputs._rows[key] if donor is None else donor
            inputs._verify_source(key)
            if donor is not None:
                source = donor_sources[key]
                if sha256_file(source['path']) != source['raw_sha256']:
                    raise ValueError('bound donor raw image changed')
            image = inputs._loader.load(row)  # validates decoded RGB against frozen map
            if donor is not None and sha256_file(source['path']) != source['raw_sha256']:
                image.close()
                raise ValueError('bound donor raw image changed during decode')
            return image, row['rgb_sha256']
        mapping, donor_sources = None, {}
        if command == 'potency':
            bind_acquisition_source(launch, inputs)
            from ..runtime.data_io import load_normlens
            from ..data.eval_manifests import shuffle_maps
            verify_sidecar(launch['shuffle_map'])
            mapping = json.loads(Path(launch['shuffle_map']).read_text())
            rows = load_normlens(data['normlens_manifest'], data['normlens_manifest_sha256'])
            frozen = shuffle_maps(rows)['dev']['mapping']
            if mapping != frozen:
                raise ValueError('shuffle map is not frozen Appendix A dev mapping')
            validate_index(launch['cache_index'], inputs, pins=pins, alignment_ids=alignment)
            donor_sources = donor_source_pins(mapping, rows, data['normlens_image_dir'])
        budget.check()
        if command == 'profile-preseal':
            from .profiling import execute_preseal
            return execute_preseal(serializer, launch, inputs, pins=pins, alignment_ids=alignment,
                runtime_pins=runtime_pins, budget=budget)
        model, _ = load_model(paths['model_dir'], runtime_pins)
        budget.check()
        if command == 'eval':
            from .evaluation import execute_loaded
            return execute_loaded(model, serializer, launch, pins=pins, budget=budget)
        if command == 'export':
            from .export_pipeline import execute_loaded
            return execute_loaded(model, serializer, launch, inputs, pins=pins, alignment_ids=alignment, budget=budget)
        hook = model.register_forward_pre_hook(lambda *unused: budget.check())
        if command in ('train','c2'):
            from .training import execute_loaded
            return execute_loaded(model, serializer, launch, inputs, pins=pins, alignment_ids=alignment,
                                  budget=budget, resume=resume, c2_only=command == 'c2')
        if command == 'cache':
            path, state = cache_inputs(model, serializer, inputs, paths['output_root'], pins=pins,
                alignment_ids=alignment, image_for=image_for, budget=budget)
            return dict(index=str(path), complete=state['complete'], coverage=state['coverage'])
        if launch.get('diagnostic_amendment') is not None:
            from .diagnostic_potency import execute_loaded
            return execute_loaded(model, launch, inputs, pins=pins, alignment_ids=alignment, budget=budget)
        return probe(model, serializer, inputs, launch['cache_index'], mapping, paths['output_root'],
            pins=pins, alignment_ids=alignment, image_for=image_for, budget=budget, attempt_id=auth['reservation_id'], donor_sources=donor_sources)
    except BudgetExpired:
        write_once_json(Path(paths['output_root']) / ('attempt-' + auth['reservation_id'] + '-incomplete.json'),
            dict(status='INCOMPLETE', reason='walltime exhausted', phase=phase, intervention_training_allowed=False))
        raise
    finally:
        if hook is not None:
            hook.remove()
