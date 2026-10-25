"""CPU-only identity checks for reports consuming native hybrid evaluation artifacts."""
from pathlib import Path
import re
from .constants import REMOTE_ROOT
from .hashing import sha256_file
from .io_utils import read_json, verify_sidecar

HYBRID_ARMS = ('H-T', 'H-R', 'F-T', 'D-T')
INFERENCE_LIMITATION = ('Saved-export provenance does not prove the adapter dtype or tensor equality used by the actual evaluation forward. These inputs contain no loaded-adapter verification receipt; artifact diagnostics must not certify verified inference.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _consumer(value):
    require(isinstance(value, dict) and set(value) == {
        'code_root','source_snapshot_sha256','source_manifest_sha256'}, 'hybrid consumer identity incomplete')
    require(_sha(value['source_snapshot_sha256']) and _sha(value['source_manifest_sha256']) and
        value['code_root'] == str(Path(REMOTE_ROOT)/'releases'/value['source_snapshot_sha256']),
        'hybrid consumer source identity mismatch')


def _loaded_adapter_proof(identity, exported):
    proof=identity.get('loaded_adapter_verification')
    if proof is None:
        return None
    required=dict(schema='gn_hybrid_loaded_adapter_v1',checked_before_inference=True,
        loader_autocast_adapter_dtype=True,saved_dtype='torch.float32',loaded_dtype='torch.float32',
        finite=True,bitwise_equal=True,export_report_sha256=identity['export_report_sha256'],
        adapter_config_sha256=identity['adapter_config_sha256'],checkpoint_sha256=identity['checkpoint_sha256'],
        checkpoint_adapter_sha256=exported.get('adapter_sha256'))
    require(isinstance(proof, dict) and all(proof.get(k)==v for k,v in required.items()),
        'invalid loaded-adapter verification proof')
    require(all(proof.get(k) is True for k in ('checked_before_inference','loader_autocast_adapter_dtype',
        'finite','bitwise_equal')), 'loaded-adapter proof flags must be explicit booleans')
    require(type(proof.get('tensor_count')) is int and proof['tensor_count']>0 and
        all(_sha(proof.get(k)) for k in ('saved_tensor_sha256','loaded_tensor_sha256',
            'checkpoint_adapter_sha256','adapter_file_sha256')) and
        proof['saved_tensor_sha256']==proof['loaded_tensor_sha256'],
        'loaded-adapter proof tensor identity mismatch')
    before=proof.get('base_parameter_dtype_counts_before')
    require(isinstance(before, dict) and before and all(type(v) is int and v>0 for v in before.values()) and
        before==proof.get('base_parameter_dtype_counts_after'), 'loaded-adapter proof base dtype mismatch')
    return proof


def validate_hybrid_identity(identity, *, arm, task, condition, export_report, scoring_seal_sha256):
    """Validate the actual hybrid schema, without renaming arms or baseline fields.

    The hybrid exporter intentionally writes no report sidecar. Its exact bytes
    are instead bound by the already sidecar-verified evaluation identity.
    """
    fields = {'schema','phase','arm','seed','task','split','condition','model_pins',
        'scoring_seal_sha256','seal_a_sha256','seal_b_sha256','operator_plan_sha256',
        'trained_source_snapshot_sha256','export_report_sha256','checkpoint_sha256',
        'adapter_config_sha256','export_consumer','eval_consumer','data_manifest_sha256',
        'expected_ids_sha256','answer_letters','shuffle_map_sha256'}
    require(fields <= identity.keys(), 'incomplete hybrid provenance identity')
    require(identity['schema'] == 'gn_hybrid_eval_v1' and identity['phase'] == 'hybrid_eval' and
        arm in HYBRID_ARMS and identity['arm'] == arm and type(identity['seed']) is int and
        identity['seed'] in (17,29,43) and identity['task'] == task and
        identity['condition'] == condition and identity['split'] == 'test', 'hybrid run identity mismatch')
    for key in ('scoring_seal_sha256','seal_a_sha256','seal_b_sha256','operator_plan_sha256',
                'trained_source_snapshot_sha256','export_report_sha256','checkpoint_sha256',
                'adapter_config_sha256','data_manifest_sha256','expected_ids_sha256'):
        require(_sha(identity[key]), 'hybrid provenance SHA256 missing: '+key)
    require(identity['scoring_seal_sha256'] == scoring_seal_sha256, 'hybrid scoring seal mismatch')
    for key in ('export_consumer','eval_consumer'):
        _consumer(identity[key])
    require(isinstance(identity['model_pins'], dict) and set(identity['model_pins']) == {
        'model_sha256','processor_sha256','template_sha256'} and
        all(_sha(v) for v in identity['model_pins'].values()), 'hybrid model pins incomplete')
    require(export_report is not None, 'hybrid report requires --export-report')
    path=Path(export_report)
    report_sha=sha256_file(path)
    require(report_sha == identity['export_report_sha256'], 'hybrid export report binding mismatch')
    side=path.with_name(path.name+'.sha256')
    if side.exists():
        verify_sidecar(path)
    report=read_json(path)
    exported=report.get('identity', {})
    expected=dict(arm=arm,seed=str(identity['seed']),checkpoint_sha256=identity['checkpoint_sha256'],
        seal_b=identity['seal_b_sha256'],protocol=identity['operator_plan_sha256'],
        source_snapshot=identity['trained_source_snapshot_sha256'],
        **{key.removesuffix('_sha256'):value for key,value in identity['model_pins'].items()},
        **{'export_consumer_'+key:value for key,value in identity['export_consumer'].items()})
    require(report.get('passed') is True and all(exported.get(k)==v for k,v in expected.items()),
        'hybrid export checkpoint/Seal B/source/model identity mismatch')
    require(len(report.get('reloads', [])) >= 2 and all(r.get('passed') is True and
        r.get('adapter', {}).get('bitwise_equal') is True for r in report['reloads']),
        'hybrid export lacks two verified adapter reloads')
    config=path.parent/'adapter_config.json'
    require(sha256_file(config)==identity['adapter_config_sha256'], 'hybrid adapter config binding mismatch')
    loaded_proof=_loaded_adapter_proof(identity, exported)
    return dict(path=str(path.resolve()),sha256=report_sha,
        sidecar_sha256=sha256_file(side) if side.exists() else None,
        binding='exact export bytes and sibling config bound by evaluation identity',
        provenance_verified_saved_export=True, inference_adapter_identity_unverified=loaded_proof is None,
        loaded_adapter_verification=loaded_proof)


def validate_hybrid_ego_identity(identity, *, export_report):
    """Exact native hybrid transfer provenance; no baseline schema substitution."""
    require(identity.get('schema')=='gn_hybrid_ego_eval_v1' and
        identity.get('phase')=='hybrid_eval' and identity.get('arm') in HYBRID_ARMS and
        type(identity.get('seed')) is int and identity['seed'] in (17,29,43),
        'explicit hybrid Ego identity required')
    for key in ('source_sha256','ids_manifest_sha256','sheet_manifest_sha256','pins_sha256',
                'seal_a_sha256','transfer_seal_sha256','seal_b_sha256','operator_plan_sha256',
                'trained_source_snapshot_sha256','adapter_export_report_sha256',
                'adapter_config_sha256','checkpoint_sha256'):
        require(_sha(identity.get(key)), 'hybrid Ego SHA256 missing: '+key)
    from .prompts import verify_ego_template_hashes
    from .runtime.gates import EGO_MAX_PROMPT_RANGE
    require(identity.get('max_new_tokens')==256 and
        identity.get('template_sha256')==verify_ego_template_hashes() and
        type(identity.get('max_prompt_tokens')) is int and
        EGO_MAX_PROMPT_RANGE[0]<=identity['max_prompt_tokens']<=EGO_MAX_PROMPT_RANGE[1],
        'hybrid Ego native cap/templates required')
    for key in ('export_consumer','eval_consumer'):
        _consumer(identity.get(key))
    path=Path(export_report)
    require(sha256_file(path)==identity['adapter_export_report_sha256'], 'hybrid Ego export bytes mismatch')
    report=read_json(path);exported=report.get('identity',{})
    expected=dict(arm=identity['arm'],seed=str(identity['seed']),
        checkpoint_sha256=identity['checkpoint_sha256'],seal_b=identity['seal_b_sha256'],
        protocol=identity['operator_plan_sha256'],source_snapshot=identity['trained_source_snapshot_sha256'])
    require(report.get('passed') is True and all(exported.get(k)==v for k,v in expected.items()),
            'hybrid Ego selected export identity mismatch')
    require(all(exported.get('export_consumer_'+k)==v for k,v in identity['export_consumer'].items()),
            'hybrid Ego export consumer mismatch')
    require(len(report.get('reloads',[]))==2 and all(r.get('passed') is True and
        r.get('adapter',{}).get('bitwise_equal') is True for r in report['reloads']),
        'hybrid Ego export requires two verified native reloads')
    pins=identity.get('model_pins',{})
    require(set(pins)=={'model_sha256','processor_sha256','template_sha256'} and
        all(_sha(v) for v in pins.values()) and all(exported.get(k)==pins[k+'_sha256']
            for k in ('model','processor','template')), 'hybrid Ego model pins mismatch')
    require(sha256_file(path.parent/'adapter_config.json')==identity['adapter_config_sha256'],
            'hybrid Ego adapter configuration mismatch')
    # This shared guard expects the standard export-report key, without changing
    # the Ego identity/schema or any predictions used for scoring.
    proof_identity={**identity,'export_report_sha256':identity['adapter_export_report_sha256']}
    require(_loaded_adapter_proof(proof_identity,exported) is not None,
            'hybrid Ego scientific report requires loaded FP32 proof')
    return report
