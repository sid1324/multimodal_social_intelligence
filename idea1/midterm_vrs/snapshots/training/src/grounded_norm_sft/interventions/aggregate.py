"""Explicit aggregate authority with audited, append-only predecessor continuation.

This module creates no consent and never modifies predecessor evidence or ledgers.
"""
from contextlib import ExitStack
from pathlib import Path

from ..hashing import sha256_file
from ..runtime import ledger
from ..runtime.errors import GateError


def fail(message):
    raise GateError(['GATE aggregate approval: ' + message])


def ref(value):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str):
        fail('path/SHA256 evidence required')
    path = Path(value['path'])
    if not path.is_file() or sha256_file(path) != value.get('sha256'):
        fail('evidence missing or SHA256 mismatch: ' + str(path))
    return path


def validate_authority(a):
    """Called by primitive envelope loader; no recursive predecessor loading."""
    problems = []
    if (a.get('total_su') != 200 or type(a.get('total_su')) not in (int, float) or
            a.get('total_gpu_hours') != 100 or type(a.get('total_gpu_hours')) not in (int, float) or
            a.get('su_per_gpu_hour') != 2 or type(a.get('su_per_gpu_hour')) not in (int, float) or
            a.get('max_job_minutes') != 360 or type(a.get('max_job_minutes')) is not int or
            type(a.get('max_concurrent_gpus')) is not int or not 1 <= a['max_concurrent_gpus'] <= 3):
        problems.append('v3 requires 100 GPU-hours / 200 SU, verified rate 2, six-hour jobs and 1..3 concurrent one-GPU jobs')
    source = ref(a.get('source_approval')).read_text()
    for quote in ("GPU hours, I'm giving you permission for that. Just run everything, just submit all the jobs. And if there are some problems and we need to change it, we will make the change as these tests are going on. I know this is a sequential thing, but submit everything",
                  'Or hundred GPU hour, I just want it to be done by tomorrow'):
        if quote not in source:
            problems.append('source receipt must contain exact trusted aggregate authorization')
    ref(a.get('predecessor_registry'))
    scope = ledger._strict_load(ref(a.get('scope_approval')))
    required = dict(schema='gn_aggregate_scope_approval_v1', approved=True, approved_by='user',
                    approval_id=a.get('approval_id'), total_gpu_hours=100, total_su=200,
                    max_concurrent_gpus=a.get('max_concurrent_gpus'),
                    registry_sha256=a['predecessor_registry']['sha256'],
                    source_approval_sha256=a['source_approval']['sha256'])
    if (not isinstance(scope, dict) or any(scope.get(k) != v for k, v in required.items()) or
            type(scope.get('approved')) is not bool or
            sorted(scope.get('phases', [])) != sorted(a.get('phases', {})) or
            not isinstance(scope.get('quote'), str) or not scope['quote'].strip() or scope['quote'] not in source):
        problems.append('scope receipt must bind exact authority, source, complete predecessor registry and phases')
    phases = a.get('phases')
    for spec in phases.values() if isinstance(phases, dict) else ():
        if isinstance(spec, dict) and ledger.finite_positive(spec.get('max_wall_minutes')) and spec['max_wall_minutes'] > 360:
            problems.append('phase wall time exceeds six hours')
    return problems


def predecessors(a, phases):
    registry = ledger._strict_load(ref(a.get('predecessor_registry')))
    entries = registry.get('entries') if isinstance(registry, dict) else None
    if not isinstance(registry, dict) or registry.get('schema') != 'gn_aggregate_predecessors_v1' or not isinstance(entries, list) or not entries:
        fail('nonempty exact predecessor registry required')
    seen_ids, seen_paths = {a['approval_id']}, {Path(a['ledger_path']).resolve()}
    result = []
    for entry in entries:
        ep = ref(entry.get('approval'))
        old = ledger.load_approval(ep, entry['approval']['sha256'], phases)
        if not ledger.is_v2(old):
            fail('predecessors must be immutable v2 envelopes')
        lp = Path(old['ledger_path'])
        if old['approval_id'] in seen_ids or lp.resolve() in seen_paths:
            fail('duplicate predecessor identity or ledger (double counting refused)')
        seen_ids.add(old['approval_id']); seen_paths.add(lp.resolve())
        snapshot = ref(entry.get('ledger_snapshot')).read_bytes()
        live = lp.read_bytes()
        if not snapshot.endswith(b'\n') or not live.startswith(snapshot):
            fail('predecessor ledger truncated or snapshot prefix rewritten')
        snapshot_events = ledger.read_events(entry['ledger_snapshot']['path'])
        initial = ledger.accounting(snapshot_events, old['approval_id'], entry['approval']['sha256'], strict=True)
        if entry.get('status') == 'terminal':
            if initial['open_reservations'] or live != snapshot:
                fail('terminal predecessor must stay fully settled and frozen')
        elif entry.get('status') == 'active':
            # Existing reservations can finish; superseded authority cannot add new work.
            for event in ledger.read_events(lp)[len(snapshot_events):]:
                if event.get('event') == 'reserve' or event.get('reservation_id') not in initial['reservations']:
                    fail('active predecessor continuation added unauthorized reservations')
        else:
            fail('predecessor status must be terminal or active')
        current = ledger.ledger_accounting(old, entry['approval']['sha256'])
        result.append((old, current))
    return result


def load_approval(path, sha, phases):
    a = ledger.load_approval(path, sha, phases)
    a['_envelope_path'] = str(Path(path).resolve())
    phases = a['_aggregate_phases']
    prior = predecessors(a, phases)
    current = ledger.ledger_accounting(a, sha)
    if sum(acct['committed_total'] for _, acct in prior) + current['committed_total'] > a['total_su'] + ledger.SU_EPS:
        fail('historical actual plus open reservations exceed aggregate ceiling')
    return a


def locked(a):
    stack = ExitStack()
    try:
        paths = [Path(a['ledger_path'])] + [Path(old['ledger_path']) for old, _ in predecessors(a, a['_aggregate_phases'])]
        for path in sorted({p.resolve() for p in paths}, key=str):
            stack.enter_context(ledger._locked(path))
    except BaseException:
        stack.close(); raise
    return stack


def reserve(a, sha, phase, minutes, *, note='', after_job=None):
    with locked(a):
        fresh = load_approval(a['_envelope_path'], sha, a['_aggregate_phases'])
        prior = sum(acct['committed_total'] for _, acct in predecessors(fresh, a['_aggregate_phases']))
        # All ledger locks are already held; append without reacquiring flock.
        return _reserve_locked(fresh, sha, phase, minutes, prior, note, after_job)


def _reserve_locked(a, sha, phase, minutes, prior, note, after_job):
    if phase not in a['phases'] or not ledger.finite_positive(minutes) or minutes > min(360, a['phases'][phase]['max_wall_minutes']):
        fail('phase or wall time outside explicit authority')
    acct = ledger.ledger_accounting(a, sha)
    su = ledger.job_su(a, minutes)
    if acct['committed_total'] + prior + su > a['total_su'] + ledger.SU_EPS:
        fail('reservation exceeds aggregate actual-plus-reserved budget')
    if acct['committed_by_phase'].get(phase, 0) + su > a['phases'][phase]['max_su'] + ledger.SU_EPS:
        fail('reservation exceeds phase budget')
    if after_job is not None and not ledger.valid_job_id(after_job):
        fail('invalid predecessor allocation identity')
    import uuid
    event = ledger._event('reserve', a, uuid.uuid4().hex, approval_sha256=sha, phase=phase,
                          time_limit_minutes=minutes, su=su, note=note, after_job=after_job)
    ledger._append(Path(a['ledger_path']), event)
    return event


def mark_started(a, sha, rid, phase, minutes, job_id):
    with locked(a):
        fresh = load_approval(a['_envelope_path'], sha, a['_aggregate_phases'])
        problems = ledger.check_reservation(fresh, sha, rid, phase, minutes,
            ledger.ledger_accounting(fresh, sha)['reservations'].get(rid, {}).get('after_job'))
        accts = [acct for _, acct in predecessors(fresh, a['_aggregate_phases'])] + [ledger.ledger_accounting(fresh, sha)]
        running = 0
        for acct in accts:
            for other_id, r in acct['reservations'].items():
                if other_id == rid:
                    submission = (r.get('submission') or {}).get('slurm_job_id')
                    if submission != job_id:
                        problems.append('GATE aggregate queue: runtime allocation must match submission')
                    if r.get('after_job') and not ledger.predecessor_terminal(r['after_job']):
                        problems.append('GATE aggregate queue: predecessor not positively terminal')
                elif r['consumed_su'] is None and r['started'] is not None:
                    old_job = (r.get('submission') or r['started']).get('slurm_job_id')
                    if not old_job or not ledger.predecessor_terminal(old_job):
                        running += 1
        if running >= fresh['max_concurrent_gpus']:
            problems.append('GATE aggregate queue: lineage GPU concurrency ceiling reached')
        if problems:
            raise GateError(problems)
        event = ledger._event('start', fresh, rid, slurm_job_id=job_id)
        ledger._append(Path(fresh['ledger_path']), event)
        return event


def cumulative_accounting(a, sha):
    current = ledger.ledger_accounting(a, sha)
    prior = sum(acct['committed_total'] for _, acct in predecessors(a, a['_aggregate_phases']))
    total = prior + current['committed_total']
    rate = a['su_per_gpu_hour']
    return dict(current, predecessor_committed_su=prior, cumulative_committed_su=total,
                cumulative_cap_su=a['total_su'], cumulative_remaining_su=a['total_su']-total,
                predecessor_committed_gpu_hours=prior/rate, cumulative_committed_gpu_hours=total/rate,
                cumulative_cap_gpu_hours=a['total_su']/rate,
                cumulative_remaining_gpu_hours=(a['total_su']-total)/rate)
