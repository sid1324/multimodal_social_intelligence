"""Explicit prospective eta=.5 F-T/D-T capability; never an execution grant."""
from pathlib import Path
from ..hashing import sha256_file, sha256_bytes, compact_json_bytes
from ..io_utils import read_json, verify_sidecar
from .cache import digest
from .eta050 import Eta050Protocol, _VALIDATED, require_dose

PROTOCOL = 'prospective-srer-family-controls-eta050-v1'
APPROVAL_SHA256 = '2dec703fab6a7df13bc0e7b213c070330ec18b8fb7e46332fd657bda6291d366'
QUOTE = 'Approve η=0.5 family controls'
ARMS = ('F-T', 'D-T')
SUPPORT_POLICY = 'targeted_cached_support_with_matched_clean_random_c2'
SHARED_PROOF = 'shared_both_family_engineering_proof'


def validate_family_protocol(parent, spec):
    require_dose(.5, parent)
    identity = parent.as_dict()
    if identity.get('family_control_protocol') is not None:
        raise ValueError('original H-T/H-R parent amendment required')
    if (not isinstance(spec, dict) or spec.get('protocol') != PROTOCOL or spec.get('eta') != .5 or
            spec.get('arms') != list(ARMS) or spec.get('support_policy') != SUPPORT_POLICY or
            spec.get('parent_amendment_sha256') != identity['amendment_sha256']):
        raise ValueError('explicit F-T/D-T family protocol bound to original amendment required')
    approval = spec.get('user_approval') or {}
    path = Path(approval.get('path') or '')
    if (not path.is_file() or approval.get('sha256') != APPROVAL_SHA256 or
            sha256_file(path) != APPROVAL_SHA256 or QUOTE not in path.read_text()):
        raise ValueError('exact prospective family-control approval bytes/quote required')
    identity.update(family_control_protocol=PROTOCOL, family_control_approval_sha256=APPROVAL_SHA256,
                    family_control_amendment_sha256=digest(spec))
    return Eta050Protocol(tuple(sorted(identity.items())), _VALIDATED)


def is_family(protocol):
    return protocol is not None and protocol.as_dict().get('family_control_protocol') == PROTOCOL


def parent_identity(protocol):
    return {k:v for k,v in protocol.as_dict().items() if not k.startswith('family_control_')}


def validate_family_evidence(launch, seal, artifacts, protocol, schedules, pins_sha256):
    """Main training needs actual paired updates and seed-matched clean initialization."""
    extra = {}
    for key in ('family_profile', 'matched_clean'):
        ref = seal['artifacts'][key]
        verify_sidecar(ref['path'])
        if sha256_file(ref['path']) != ref['sha256']:
            raise ValueError('family evidence bytes mismatch: '+key)
        extra[key] = read_json(ref['path'], verify=True)
    profile, clean = extra['family_profile'], extra['matched_clean']['identity']
    expected = dict(schema='gn_hybrid_preseal_update_profile_v1', status='COMPLETE',
        seed=launch['seed'], epochs=3, selected_core_arms=list(ARMS), dose_protocol=protocol.as_dict(),
        source_snapshot_sha256=launch['authorization']['source_snapshot_sha256'],
        source_manifest_sha256=launch['authorization']['source_manifest_sha256'],
        runtime_pins_sha256=pins_sha256,
        train_cache_index_sha256=launch['train_cache_index_sha256'],
        dev_cache_index_sha256=launch['dev_cache_index_sha256'])
    if any(profile.get(k) != v for k,v in expected.items()) or set(profile.get('arms', {})) != set(ARMS):
        raise ValueError('complete exact-source paired F-T/D-T family profile required')
    if profile.get('native_acceptance') != seal['artifacts']['acceptance']:
        raise ValueError('family profile must bind the actual shared native proof')
    acceptance = artifacts['acceptance']
    if (acceptance.get('dose_protocol') != parent_identity(protocol) or
            acceptance.get('native_acceptance_role') != SHARED_PROOF or
            acceptance.get('family_profile_protocol_sha256') != digest(protocol.as_dict())):
        raise ValueError('shared native proof must bind original dose and exact family profile capability')
    clean_schedule = schedules['B1']
    clean_expected = dict(arm='B1', seed=launch['seed'], epochs=3, pins_sha256=pins_sha256,
        normlens_manifest_sha256=launch['data']['normlens_manifest_sha256'],
        normbank_manifest_sha256=launch['data']['normbank_manifest_sha256'],
        schedule_sha256=sha256_bytes(compact_json_bytes(clean_schedule)))
    if any(clean.get(k) != v for k,v in clean_expected.items()) or not clean.get('lora_init_sha256'):
        raise ValueError('exact seed-matched B1 clean exposure and initialization receipt required')
    for arm in ARMS:
        measured = profile['arms'][arm]
        init = measured.get('initialization') or {}
        if (measured.get('arm') != arm or measured.get('seed') != launch['seed'] or
                measured.get('status') != 'COMPLETE' or measured.get('completed_updates', 0) < 2 or
                measured.get('schedule_sha256') != digest(schedules[arm]) or
                init.get('adapter_init') != clean['lora_init_sha256'] or
                init.get('frozen_model') != extra['matched_clean'].get('frozen_state')):
            raise ValueError('actual two-update family route profile with matched clean initialization required')
        feasibility = artifacts['runtime_feasibility']['arms'][arm]
        if seal['artifacts']['family_profile'] not in feasibility['measurement_artifacts']:
            raise ValueError('family feasibility must cite actual family update measurements')
    return extra
