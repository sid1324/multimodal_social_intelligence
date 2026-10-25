"""Read-only hybrid launch gates. Never creates approvals or reservations."""
import datetime
from pathlib import Path
from ..hashing import sha256_file
from ..runtime import gates as baseline
from ..runtime.errors import GateError
from ..runtime import ledger
from . import approval as hybrid_approval
from ..runtime.pins import load_pins, open_pin_gates, verify_model_dir, verify_processor_dir
from ..runtime.source_snapshot import verify_source_snapshot

SCHEMA = 'gn_hybrid_launch_v1'
PHASES = ('hybrid_cache', 'hybrid_potency', 'hybrid_train', 'hybrid_c2', 'hybrid_export', 'hybrid_profile_preseal', 'hybrid_eval')


def check_launch(launch, phase, *, env=None, audit_assets=True, require_reservation=True):
    pins, pins_sha = load_pins()
    c = baseline._Collector()
    if phase not in PHASES or launch.get('phase') != phase:
        c.add('phase', 'explicit authorized hybrid phase required')
    if launch.get('schema') != SCHEMA:
        c.add('launch', 'schema must be ' + SCHEMA)
    auth, paths = launch.get('authorization', {}), launch.get('paths', {})
    approval = ledger.load_approval(auth.get('approval_path', ''), auth.get('approval_sha256', ''),
                                    baseline.PHASES + PHASES, pins['psc'].get('approved_su_budget'))
    if phase in approval['phases']:
        approval = hybrid_approval.load_approval(auth['approval_path'], auth['approval_sha256'],
            baseline.PHASES + PHASES, pins['psc'].get('approved_su_budget'))
    spec = approval['phases'].get(phase)
    if spec is None:
        c.add('approval', 'envelope does not explicitly authorize ' + phase)
    baseline._job(c, launch, pins, phase)
    minutes = launch.get('job', {}).get('time_limit_minutes')
    reserve = launch.get('job', {}).get('item_reserve_seconds')
    margin = launch.get('job', {}).get('stop_margin_minutes')
    if not ledger.finite_positive(reserve) or (ledger.finite_positive(minutes) and ledger.finite_positive(margin)
            and reserve >= (minutes - margin) * 60):
        c.add('job', 'item_reserve_seconds must be explicit finite positive and fit inside time minus stop margin')
    if spec is not None and ledger.finite_positive(minutes):
        if minutes > spec['max_wall_minutes'] or ledger.job_su(approval, minutes) > spec['max_su']:
            c.add('approval', 'job exceeds explicit phase time/SU ceiling')
    if require_reservation:
        c.problems.extend(ledger.check_reservation(approval, auth.get('approval_sha256'),
            auth.get('reservation_id'), phase, minutes, auth.get('after_job')))
    for pin in open_pin_gates(pins):
        if pin != 'psc.approved_su_budget' or not (ledger.is_v2(approval) or ledger.is_v3(approval)):
            c.add('pins', pin + ' is open')
    if env is not None:
        c.problems.extend(baseline.check_environment(launch, env))
        if env.get('SLURM_GPUS_ON_NODE') != '1':
            c.add('environment', 'explicit SLURM_GPUS_ON_NODE=1 required')
    # Refuse unauthorized jobs before expensive asset hashing or model imports.
    if c.problems:
        raise GateError(c.problems)
    root = paths.get('code_root')
    if not root or not auth.get('source_snapshot_sha256') or not auth.get('source_manifest_sha256'):
        c.add('source snapshot', 'code_root, snapshot and manifest SHA256 required')
    else:
        verify_source_snapshot(root, auth['source_snapshot_sha256'])
        baseline._check_file_hash(c, 'source snapshot', str(Path(root) / 'SHA256SUMS'), auth['source_manifest_sha256'], True)
        from .export_lineage import execution_root, verify_export_consumer
        execution=execution_root(launch)
        try:
            if phase=='hybrid_eval':
                from .eval_lineage import verify_eval_consumer, execution_root as eval_root
                verify_eval_consumer(launch)
                execution=eval_root(launch)
            else:
                verify_export_consumer(launch)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            c.add('export consumer', str(exc))
        if env is not None and Path(__file__).resolve().parents[3] != Path(execution).resolve():
            c.add('source snapshot', 'interventions imported outside bound execution code_root')
    baseline._rights(c, launch, ['normlens', 'normbank'] if phase in ('hybrid_train','hybrid_c2','hybrid_export','hybrid_profile_preseal','hybrid_eval') else ['normlens'], datetime.datetime.now(datetime.timezone.utc))
    hashes = baseline._data(c, launch, True, phase in ('hybrid_train','hybrid_c2','hybrid_export','hybrid_profile_preseal','hybrid_eval'), True)
    baseline._seal_a(c, launch, dict(hashes, runtime_pins=pins_sha), True)
    if phase != 'hybrid_eval' and launch.get('split') not in ('train', 'dev') or phase == 'hybrid_potency' and launch.get('split') != 'dev':
        c.add('data', 'cache split must be train/dev; potency must be dev')
    plan = launch.get('operator_plan', {})
    baseline._check_file_hash(c, 'operator_plan', plan.get('path'), plan.get('sha256'), True)
    frozen_plan = Path(__file__).resolve().parents[3] / 'provenance/hybrid_norm_sft_plan_FINAL_v1.2.md'
    if plan.get('sha256') != sha256_file(frozen_plan):
        c.add('operator_plan', 'must bind the frozen v1.2 plan including9C/SealB')
    if phase == 'hybrid_potency':
        from .acquisition_source import verify_launch_acquisition
        try:
            verify_launch_acquisition(launch)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            c.add('cache acquisition source', str(exc))
        baseline._check_file_hash(c, 'cache index', launch.get('cache_index'), launch.get('cache_index_sha256'), True)
        for key in ('shuffle_map',):
            baseline._check_file_hash(c, key, launch.get(key), launch.get(key + '_sha256'), True)
    if phase in ('hybrid_train','hybrid_c2','hybrid_export'):
        if launch.get('split') != 'train':
            c.add('data', 'hybrid training/C2 requires train source split')
        try:
            validate_seal_b(launch, pins_sha)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
            c.add('seal_b', str(exc))
        for key in ('update_reserve_seconds','dev_reserve_seconds'):
            if not ledger.finite_positive(launch.get('job', {}).get(key)):
                c.add('job', key + ' must be measured explicit finite positive')
    if phase == 'hybrid_eval':
        baseline._eval_spec(c, launch, phase, True)
        if launch.get('eval', {}).get('split') != 'test' or launch.get('split') != 'train':
            c.add('eval', 'hybrid evaluation requires test split and train source for Seal B')
        try:
            validate_seal_b(launch, pins_sha)
            from .evaluation import verify_export, evaluation_cohort, validate_scoring_seal
            validate_scoring_seal(launch)
            evaluation_cohort(launch)
            from .cache import digest
            from ..hashing import sha256_bytes
            from ..runtime.pins import receipt_chat_template
            # Match the exact identities used by the hybrid exporter.
            export_pins = dict(model_sha256=digest(pins['model']['safetensors_shard_sha256']),
                processor_sha256=pins['model']['processor_and_chat_template_sha256'],
                template_sha256=sha256_bytes(receipt_chat_template(pins).encode()))
            verify_export(launch, export_pins)
            seconds = launch['eval']['first_forward_reserve_minutes'] * 60
            if not ledger.finite_positive(seconds) or seconds >= 60*(minutes-margin):
                raise ValueError('first-forward reserve must fit usable job time')
        except (KeyError, ValueError, OSError, RuntimeError, TypeError) as exc:
            c.add('hybrid_eval', str(exc))
    if phase == 'hybrid_profile_preseal':
        try:
            from .profiling import validate_profile_launch
            validate_profile_launch(launch)
        except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
            c.add('profile_preseal', str(exc))
    if phase == 'hybrid_export':
        from ..io_utils import read_json
        import math
        export = launch.get('export', {})
        seconds = export.get('child_timeout_seconds')
        ids = export.get('fixture_ids')
        if (type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds <= 0 or
                not isinstance(ids,list) or not ids or any(not isinstance(k,str) or not k for k in ids) or len(set(ids)) != len(ids)):
            c.add('export', 'positive child_timeout_seconds and unique fixed dev fixture_ids required')
        if not paths.get('processor_dir'):
            c.add('export', 'explicit pinned processor_dir required')
        try:
            from ..runtime.checkpoint import verify_checkpoint_dir
            selection = read_json(Path(paths['training_run_dir']) / 'selection.json', verify=True)
            verify_checkpoint_dir(selection['checkpoint']['dir'], paths['training_run_dir'])
            baseline._check_file_hash(c, 'selected checkpoint', str(Path(selection['checkpoint']['dir']) / 'MANIFEST.json'),
                                     selection['checkpoint']['manifest_sha256'], True)
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            c.add('export', 'completed selected hybrid checkpoint required: '+str(exc))
    if c.problems:
        raise GateError(c.problems)
    # CPU file audits only; load_model will repeat these before touching weights.
    if audit_assets:
        verify_model_dir(pins, paths['model_dir'])
        verify_processor_dir(pins, paths.get('processor_dir', paths['model_dir']), require_pin=True)
    return dict(phase=phase, pins_sha256=pins_sha, approved=True), approval, pins


def validate_seal_b(launch, pins_sha256):
    """Seal B is evidence-bound; arithmetic potency alone grants no authority."""
    import json
    import math
    from ..io_utils import read_json, verify_sidecar
    from .cache import digest
    from .potency import aggregate, _finite
    from ..schedules import build_training_schedule
    from ..runtime.data_io import load_normlens, load_normbank, split_rows
    spec = launch['seal_b']
    verify_sidecar(spec['path'])
    if sha256_file(spec['path']) != spec['sha256']:
        raise ValueError('Seal B bytes mismatch')
    seal = read_json(spec['path'], verify=True)
    from .eta050 import launch_protocol, require_arm
    dose_protocol = launch_protocol(launch)
    from .family_controls import is_family
    family = is_family(dose_protocol)
    schema = ('gn_hybrid_seal_b_eta050_family_v1' if family else
              'gn_hybrid_seal_b_eta050_v1' if dose_protocol else 'gn_hybrid_seal_b_v1')
    if dose_protocol is not None:
        require_arm(launch['arm'], dose_protocol)
        for arm in seal.get('selected_core_arms', []):
            require_arm(arm, dose_protocol)
        from .training_cache_source import verify_cache_sources
        verify_cache_sources(launch)
    if dose_protocol is not None and seal.get('dose_protocol') != dose_protocol.as_dict():
        raise ValueError('Seal B must bind the same eta050 amendment/dose/diagnostic')
    if seal.get('schema') != schema or seal.get('checkpoint_reduction') != 'per_family_mean32':
        raise ValueError('Seal B must freeze per_family_mean32 before H training')
    if not _finite(seal):
        raise ValueError('nonfinite Seal B fields')
    bound = dict(operator_plan_sha256=launch['operator_plan']['sha256'],
        source_snapshot_sha256=launch['authorization']['source_snapshot_sha256'],
        source_manifest_sha256=launch['authorization']['source_manifest_sha256'],
        seal_a_sha256=launch['seal_a']['sha256'], runtime_pins_sha256=pins_sha256,
        arm=launch['arm'], seed=launch['seed'], epochs=launch['horizon']['epochs'])
    if any(seal.get(k) != v for k,v in bound.items()):
        raise ValueError('Seal B operator/source/data/core-run identity mismatch')
    if launch['arm'] not in ('H-T','H-R','F-T','D-T') or (type(launch['seed']) is not int or launch['seed'] not in (17,29,43)) or launch['horizon']['epochs'] != 3:
        raise ValueError('supported arm and common B1 three-epoch horizon required; H-Rmatch refused')
    if seal.get('operator_locked') is not True or seal.get('chronology') != 'baseline_first_then_frozen_base_screen_before_hybrid_training':
        raise ValueError('Seal B must disclose actual baseline-first chronology and operator lock')
    artifacts = {}
    for key in ('train_cache','dev_cache','potency','runtime_feasibility','acceptance'):
        artifact = seal['artifacts'][key]
        verify_sidecar(artifact['path'])
        if sha256_file(artifact['path']) != artifact['sha256']:
            raise ValueError('Seal B artifact bytes mismatch: ' + key)
        artifacts[key] = read_json(artifact['path'], verify=True)
    if not _finite(artifacts):
        raise ValueError('nonfinite Seal B evidence')
    for key in ('train_cache','dev_cache'):
        if not artifacts[key].get('complete'):
            raise ValueError('Seal B requires complete train/dev caches')
        launch_key = key + '_index'
        if (launch.get(launch_key) != seal['artifacts'][key]['path'] or
                launch.get(launch_key+'_sha256') != seal['artifacts'][key]['sha256']):
            raise ValueError('launch cache differs from Seal B')
    potency = artifacts['potency']
    if dose_protocol is not None:
        if seal['artifacts']['potency']['sha256'] != dose_protocol.as_dict()['diagnostic_sha256']:
            raise ValueError('eta050 Seal B requires actual revised diagnostic')
    else:
        report = aggregate(potency)
        if report['status'] != 'PASS' or potency.get('evidence_kind') != 'supplied_frozen_base':
            raise ValueError('actual native both-family potency PASS required')
    if potency.get('cache_index_sha256') != seal['artifacts']['dev_cache']['sha256']:
        raise ValueError('potency does not bind the dev cache')
    if dose_protocol is None and (potency.get('fixture_mode') or not potency.get('donor_sources_sha256')):
        raise ValueError('fixture or unbound shuffle evidence cannot authorize training')
    d = launch['data']
    nl = split_rows(load_normlens(d['normlens_manifest'], d['normlens_manifest_sha256']), 'train')
    nb = split_rows(load_normbank(d['normbank_manifest'], d['normbank_manifest_sha256']), 'train')
    schedule = build_training_schedule([r['example_id'] for r in nl], [r['example_id'] for r in nb],
                                      launch['seed'], 3, launch['arm'])
    if seal.get('schedule_sha256') != digest(schedule):
        raise ValueError('Seal B does not bind the exact three-epoch schedule')
    feasibility = artifacts['runtime_feasibility']
    if (feasibility.get('schema') != 'gn_hybrid_runtime_feasibility_v1' or
        feasibility.get('source_snapshot_sha256') != bound['source_snapshot_sha256'] or
        feasibility.get('runtime_pins_sha256') != pins_sha256 or feasibility.get('epochs') != 3 or
        feasibility.get('contingency_fraction') != .25):
        raise ValueError('measured common-horizon feasibility with 25% contingency required')
    selected = seal['selected_core_arms']
    if (not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or
            any(arm not in ('H-T','H-R','F-T','D-T') for arm in selected) or launch['arm'] not in selected or
            feasibility.get('selected_core_arms') != selected or set(feasibility['arms']) != set(selected)):
        raise ValueError('explicit selected arms and measured common horizon for each required')
    approval = hybrid_approval.load_approval(launch['authorization']['approval_path'],
        launch['authorization']['approval_sha256'], baseline.PHASES+PHASES, load_pins()[0]['psc'].get('approved_su_budget'))
    if feasibility.get('approval_sha256') != launch['authorization']['approval_sha256']:
        raise ValueError('feasibility must bind explicit compute envelope')
    allocated = feasibility['allocated_su']
    cap = approval['phases'].get('hybrid_train', {}).get('max_su', 0)
    if not ledger.finite_positive(allocated) or allocated > cap:
        raise ValueError('feasibility allocation exceeds explicit hybrid_train phase budget')
    if feasibility['allocated_wall_seconds'] != 3600*allocated/approval['su_per_gpu_hour']:
        raise ValueError('feasibility reporting window must derive from authorized SU/rate')
    required_total = 0.
    for arm in selected:
        measured = feasibility['arms'][arm]
        arm_schedule = build_training_schedule([r['example_id'] for r in nl], [r['example_id'] for r in nb],
                                              launch['seed'], 3, arm)
        components = ('complete_update_seconds','cache_seconds','native_dev_seconds','c2_seconds','io_seconds')
        if any(type(measured.get(k)) not in (int,float) or not math.isfinite(measured[k]) or measured[k] <= 0 for k in components):
            raise ValueError('finite measured complete-update/cache/dev/C2/IO times required: '+arm)
        if measured.get('schedule_sha256') != digest(arm_schedule) or not measured.get('measurement_artifacts'):
            raise ValueError('runtime measurements must bind exact schedule and native evidence: '+arm)
        for artifact in measured['measurement_artifacts']:
            verify_sidecar(artifact['path'])
            if sha256_file(artifact['path']) != artifact['sha256']:
                raise ValueError('runtime measurement evidence changed')
        required_total += 1.25*(arm_schedule['total_updates']*measured['complete_update_seconds'] + measured['cache_seconds'] +
                               4*(measured['native_dev_seconds']+measured['c2_seconds']) + measured['io_seconds'])
    if not math.isfinite(feasibility['allocated_wall_seconds']) or required_total > feasibility['allocated_wall_seconds']:
        raise ValueError('selected common horizons do not fit measured allocation with contingency')
    if seal.get('comparison_role') != ('targeted_random_comparison' if {'H-T','H-R'}.issubset(selected) else 'feasibility_only'):
        raise ValueError('H-T without H-R must be explicitly feasibility_only')
    acceptance = artifacts['acceptance']
    if dose_protocol is not None:
        if acceptance.get('evidence_kind') != 'real_9b_native_training_acceptance':
            raise ValueError('eta050 requires real9B native gradient/backend acceptance')
        if any(a.get('dose_protocol') != dose_protocol.as_dict() for a in
               ((feasibility,) if family else (feasibility, acceptance))):
            raise ValueError('eta050 feasibility and real backend acceptance must bind same amendment/dose/artifact')
    required_checks = {'native_inventory_alignment','source_algebra','prefix_isolation','endpoint_gradients',
                       'checkpointing_sequential_equivalence','same_environment_resume','native_disabled_repeats',
                       'bf16_native_reference'}
    if (acceptance.get('source_snapshot_sha256') != bound['source_snapshot_sha256'] or
            acceptance.get('runtime_pins_sha256') != pins_sha256 or set(acceptance['checks']) != required_checks):
        raise ValueError('complete source-bound Section9A/B acceptance evidence required')
    for name, check in acceptance['checks'].items():
        if check.get('exit_code') != 0 or not check.get('command') or not check.get('measurements'):
            raise ValueError('missing measured acceptance: '+name)
        artifact = check['evidence']
        verify_sidecar(artifact['path'])
        if sha256_file(artifact['path']) != artifact['sha256']:
            raise ValueError('acceptance evidence changed: '+name)
    if family:
        from .family_controls import validate_family_evidence, ARMS
        schedules = {arm:build_training_schedule([r['example_id'] for r in nl],
            [r['example_id'] for r in nb], launch['seed'], 3, arm) for arm in (*ARMS, 'B1')}
        artifacts.update(validate_family_evidence(launch, seal, artifacts, dose_protocol, schedules, pins_sha256))
    return seal, artifacts
