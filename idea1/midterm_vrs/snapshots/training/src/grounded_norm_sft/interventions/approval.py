"""Fail-closed hybrid amendment of an exhausted/frozen baseline approval.

No envelope or consent receipt is created here. The prior ledger remains untouched;
its settled usage plus this sub-envelope can never exceed the original approval.
"""
from contextlib import ExitStack
from pathlib import Path

from ..hashing import sha256_file
from ..runtime import ledger
from ..runtime.errors import GateError
from ..runtime.gates import PHASES as BASELINE_PHASES

HYBRID_PHASES = ('hybrid_cache', 'hybrid_potency', 'hybrid_train', 'hybrid_c2',
                 'hybrid_export', 'hybrid_profile_preseal', 'hybrid_eval')


def _ref(value, label):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str) or not value['path']:
        raise GateError([f'GATE hybrid approval: {label} path/SHA256 required'])
    path = Path(value['path'])
    if not path.is_file() or sha256_file(path) != value.get('sha256'):
        raise GateError([f'GATE hybrid approval: {label} missing or SHA256 mismatch'])
    return path


def load_approval(path, expected_sha256, phases, project_cap_su=None):
    approval = ledger.load_approval(path, expected_sha256, phases, project_cap_su)
    if ledger.is_v3(approval):
        from . import aggregate
        return aggregate.load_approval(path, expected_sha256, phases)
    if not ledger.is_v2(approval):
        raise GateError(['GATE hybrid approval: v2 single-GPU/six-hour enforcement required'])
    link = approval.get('prior_approval')
    prior_path = _ref(link, 'prior_approval')
    prior = ledger.load_approval(prior_path, link['sha256'], BASELINE_PHASES, project_cap_su)
    if not ledger.is_v2(prior) or prior.get('prior_approval') is not None:
        raise GateError(['GATE hybrid approval: original v2 prior approval required; recursive amendment refused'])
    prior_ledger = Path(prior['ledger_path'])
    if not prior_ledger.is_file() or sha256_file(prior_ledger) != link.get('ledger_sha256'):
        raise GateError(['GATE hybrid approval: prior ledger changed or frozen ledger SHA256 missing'])
    prior_acct = ledger.ledger_accounting(prior, link['sha256'])
    if prior_acct['open_reservations']:
        raise GateError(['GATE hybrid approval: all prior reservations must be settled'])
    if (approval['approval_id'] == prior['approval_id'] or
            Path(approval['ledger_path']).resolve() == prior_ledger.resolve() or
            Path(path).resolve() == prior_path.resolve()):
        raise GateError(['GATE hybrid approval: distinct amendment identity, envelope and ledger required'])
    approval['_checking_path'] = str(Path(path).resolve())
    predecessors = _retired(approval, prior, link)
    remaining = prior['total_su'] - prior_acct['committed_total'] - sum(
        ledger.ledger_accounting(item, ref['sha256'])['committed_total'] for ref, item in predecessors)
    if approval['total_su'] > remaining + ledger.SU_EPS:
        raise GateError(['GATE hybrid approval: sub-envelope exceeds original cumulative remaining SU'])
    if set(approval['phases']) - set(HYBRID_PHASES):
        raise GateError(['GATE hybrid approval: amendment may authorize only explicit hybrid phases'])
    scope_path = _ref(approval.get('scope_approval'), 'scope_approval')
    if scope_path.resolve() in (prior_path.resolve(), Path(prior['source_approval']['path']).resolve()):
        raise GateError(['GATE hybrid approval: distinct explicit hybrid scope consent receipt required'])
    scope = ledger._strict_load(scope_path)
    required = dict(schema='gn_hybrid_scope_approval_v1', approved=True, approved_by='user',
                    approval_id=approval['approval_id'], prior_approval_sha256=link['sha256'],
                    total_su=approval['total_su'])
    if (not isinstance(scope, dict) or any(scope.get(k) != v for k, v in required.items()) or
            type(scope.get('approved')) is not bool or
            not isinstance(scope.get('phases'), list) or
            any(not isinstance(p, str) for p in scope['phases']) or
            not ledger.finite_positive(scope.get('total_su')) or
            sorted(scope['phases']) != sorted(approval['phases']) or
            any(not isinstance(scope.get(k), str) or not scope[k].strip() for k in ('quote', 'approved_at'))):
        raise GateError(['GATE hybrid approval: explicit user receipt must bind amendment, prior identity, phases and cap'])
    if approval.get('retirement_registry') is not None and scope.get('retirement_registry_sha256') != approval['retirement_registry']['sha256']:
        raise GateError(['GATE hybrid approval: scope must bind complete retirement registry SHA256'])
    approval['_envelope_path'] = str(Path(path).resolve())
    return approval


def _retired(approval, prior, link):
    """Flat, explicitly retired children; no recursive cap or authority reset."""
    refs = approval.get('retired_predecessors', [])
    registry_ref = approval.get('retirement_registry')
    if registry_ref is None:
        # Original cache envelopes remain readable for historical accounting only.
        if refs or set(approval['phases']) != {'hybrid_cache'}:
            raise GateError(['GATE hybrid approval: complete operator retirement registry required'])
        return []
    registry = ledger._strict_load(_ref(registry_ref, 'retirement_registry'))
    if (not isinstance(refs, list) or not isinstance(registry, dict) or
            registry.get('schema') != 'gn_hybrid_retirement_registry_v1' or
            registry.get('approved') is not True or
            not isinstance(registry.get('approved_by'), str) or not registry['approved_by'].strip() or
            registry.get('prior_approval') != link or registry.get('retired_predecessors') != refs or
            any(not isinstance(registry.get(k), str) or not registry[k].strip()
                for k in ('quote', 'approved_at'))):
        raise GateError(['GATE hybrid approval: complete operator retirement registry mismatch'])
    seen_paths = {Path(approval['_checking_path']).resolve(), Path(link['path']).resolve()}
    seen_ids = {approval['approval_id'], prior['approval_id']}
    seen_ledgers = {Path(approval['ledger_path']).resolve(), Path(prior['ledger_path']).resolve()}
    result = []
    for ref in refs:
        path = _ref(ref, 'retired predecessor')
        item = ledger.load_approval(path, ref['sha256'], BASELINE_PHASES + HYBRID_PHASES)
        if (not ledger.is_v2(item) or item.get('prior_approval') != link or
                path.resolve() in seen_paths or item['approval_id'] in seen_ids or
                Path(item['ledger_path']).resolve() in seen_ledgers):
            raise GateError(['GATE hybrid approval: distinct retired predecessor with same original root required'])
        seen_paths.add(path.resolve())
        seen_ids.add(item['approval_id'])
        seen_ledgers.add(Path(item['ledger_path']).resolve())
        lp = Path(item['ledger_path'])
        if not lp.is_file() or sha256_file(lp) != ref.get('ledger_sha256'):
            raise GateError(['GATE hybrid approval: retired predecessor ledger changed'])
        if ledger.ledger_accounting(item, ref['sha256'])['open_reservations']:
            raise GateError(['GATE hybrid approval: retired predecessor must have zero open reservations'])
        result.append((ref, item))
    return result


def _ancestor_locks(approval):
    paths = [_prior_ledger(approval)]
    for ref in approval.get('retired_predecessors', []):
        item = ledger.load_approval(ref['path'], ref['sha256'], BASELINE_PHASES + HYBRID_PHASES)
        paths.append(Path(item['ledger_path']))
    stack = ExitStack()
    try:
        for path in sorted(set(p.resolve() for p in paths), key=str):
            stack.enter_context(ledger._locked(path))
    except BaseException:
        stack.close()
        raise
    return stack


def _prior_ledger(approval):
    link = approval['prior_approval']
    prior = ledger.load_approval(link['path'], link['sha256'], BASELINE_PHASES)
    return Path(prior['ledger_path'])


def reserve(approval, approval_sha256, phase, time_limit_minutes, *, note='', after_job=None):
    if ledger.is_v3(approval):
        from . import aggregate
        return aggregate.reserve(approval, approval_sha256, phase, time_limit_minutes, note=note, after_job=after_job)
    # Lock all ancestor ledgers in canonical order, then the child primitive.
    if approval.get('retirement_registry') is None:
        raise GateError(['GATE hybrid approval: retirement registry required for new execution'])
    with _ancestor_locks(approval):
        fresh = load_approval(approval['_envelope_path'], approval_sha256, BASELINE_PHASES + HYBRID_PHASES)
        return ledger.reserve(fresh, approval_sha256, phase, time_limit_minutes, note, after_job)


def mark_started(approval, approval_sha256, reservation_id, phase, time_limit_minutes, job_id):
    if ledger.is_v3(approval):
        from . import aggregate
        return aggregate.mark_started(approval, approval_sha256, reservation_id, phase, time_limit_minutes, job_id)
    if approval.get('retirement_registry') is None:
        raise GateError(['GATE hybrid approval: retirement registry required for new execution'])
    with _ancestor_locks(approval):
        fresh = load_approval(approval['_envelope_path'], approval_sha256, BASELINE_PHASES + HYBRID_PHASES)
        return ledger.mark_started(fresh, approval_sha256, reservation_id, phase, time_limit_minutes, job_id)


def cumulative_accounting(approval, approval_sha256):
    """Report original cumulative ceiling alongside the child ledger commitment."""
    if ledger.is_v3(approval):
        from . import aggregate
        return aggregate.cumulative_accounting(approval, approval_sha256)
    link = approval['prior_approval']
    prior = ledger.load_approval(link['path'], link['sha256'], BASELINE_PHASES)
    prior_acct = ledger.ledger_accounting(prior, link['sha256'])
    current = ledger.ledger_accounting(approval, approval_sha256)
    retired = _retired(approval, prior, link)
    retired_su = sum(ledger.ledger_accounting(item, ref['sha256'])['committed_total'] for ref, item in retired)
    cumulative = prior_acct['committed_total'] + retired_su + current['committed_total']
    current['retired_predecessor_consumed_su'] = retired_su
    current.update(prior_consumed_su=prior_acct['committed_total'], cumulative_committed_su=cumulative,
                   cumulative_cap_su=prior['total_su'], cumulative_remaining_su=prior['total_su']-cumulative,
                   sub_envelope_remaining_su=approval['total_su']-current['committed_total'])
    return current
