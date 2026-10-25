"""Frozen C2 checkpoint probes and post-lock descriptive aggregation.

No loader, labels in forwards, training authorization, adaptive dose, or loss term.
Clean predictions must be retained by the caller at this exact checkpoint.
"""
from collections import Counter
from contextlib import ExitStack
import math
import torch
from .cache import digest
from .data import fingerprint
from .forward import NativeForward, validated_supports
from .native_adapters import native_site
from .potency import assignments, _answer, _r, _finite, _sha

ARMS = ('H-T', 'H-R', 'F-T', 'D-T')


def identity(*, checkpoint_sha256, seed, operator_sha256, serialized, records, schedule, dose_protocol=None):
    """Identity shared by clean-dev recording and checkpoint probing."""
    if not _sha(checkpoint_sha256) or not _sha(operator_sha256) or type(seed) is not int:
        raise ValueError('checkpoint/operator SHA256 and integer seed required')
    result = dict(checkpoint_sha256=checkpoint_sha256, seed=seed, operator_sha256=operator_sha256,
                assignments=schedule, inputs={a['example_id']: fingerprint(serialized[a['example_id']]) for a in schedule},
                caches={a['example_id']: records[a['example_id']]['sha256'] for a in schedule})
    if dose_protocol is not None:
        result['dose_protocol'] = dose_protocol.as_dict()
    return result



def record_clean(identity, records):
    """Detach real logits already produced by clean dev; performs no forward.

    records are ordered (example_id, final_row_logits, answer_token_ids) tuples.
    Existing baseline prediction files omit raw logits and are insufficient.
    """
    records = list(records)
    if [r[0] for r in records] != [a['example_id'] for a in identity['assignments']]:
        raise ValueError('exact clean dev cohort/order required')
    return dict(identity=identity, rows=[dict(example_id=key, answer_token_ids=list(ids),
                answer=_answer(logits,ids)) for key,logits,ids in records])


def run(model, serialized, *, frozen_assignments, clean_predictions, checkpoint_sha256,
        seed, operator_sha256, arm, cache_paths, pins, image_sha256, alignment_ids, dose_protocol=None):
    """Exactly 64 perturbed and 80 clean forwards; clean_predictions are reused.

    Both family sites are measured for F-T/D-T as targeted support diagnostics,
    explicitly distinct from their single-family training exposure.
    """
    if arm not in ARMS:
        raise ValueError('unsupported arm (H-Rmatch is refused)')
    schedule = frozen_assignments[:32]
    if not 64 <= len(frozen_assignments) <= 128 or len(schedule) != 32 or frozen_assignments != assignments([a['example_id'] for a in frozen_assignments]):
        raise ValueError('original frozen potency cohort/order/sites required')
    records = validated_supports(serialized, cache_paths, pins=pins, image_sha256=image_sha256, alignment_ids=alignment_ids)
    eligible = [key for key, record in records.items() if record['payload']['support']['supported']]
    if frozen_assignments != assignments(eligible):
        raise ValueError('original frozen potency cohort/cache membership mismatch')
    if any(not records[a['example_id']]['payload']['support']['supported'] for a in schedule):
        raise ValueError('frozen cohort support changed')
    expected = identity(checkpoint_sha256=checkpoint_sha256, seed=seed, operator_sha256=operator_sha256,
                        serialized=serialized, records=records, schedule=schedule, dose_protocol=dose_protocol)
    if clean_predictions.get('identity') != expected or [r['example_id'] for r in clean_predictions.get('rows', [])] != [a['example_id'] for a in schedule]:
        raise ValueError('exact checkpoint clean prediction identity/order required')
    if not _finite(clean_predictions):
        raise ValueError('nonfinite clean predictions')
    native = NativeForward(model, alignment_ids=alignment_ids)
    native.bind(serialized)
    result = dict(identity=expected, arm=arm, diagnostic_role='both-family fixed-dev support control',
                  rows=[], repeats=[], finite_path=False)
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for ordinal, (a, retained) in enumerate(zip(schedule, clean_predictions['rows'])):
                key = a['example_id']; item = serialized[key]; support = records[key]['payload']['support']
                clean = retained['answer']
                _validate_answer(clean)
                probes = {}
                for family in ('FA', 'D'):
                    diagnostics = []
                    native.diagnostic = diagnostics.append
                    with native.perturbed(item, support, dict(site_layer=a[family+'_layer']),
                                          arm='H-R' if arm == 'H-R' else 'H-T', eta=.5 if dose_protocol else .25, protocol=dose_protocol):
                        logits, ids = native(item.task, item)
                    if list(ids) != retained['answer_token_ids'] or len(diagnostics) != 1:
                        raise ValueError('answer IDs or endpoint diagnostic mismatch')
                    probes[family] = dict(answer=_answer(logits, ids, clean), endpoint=diagnostics[0])
                result['rows'].append(dict(example_id=key, assignment=a, clean=clean, probes=probes))
                if ordinal >= 4:
                    continue
                reference = None
                for mode in ('untouched', 'disabled'):
                    for repeat in range(10):
                        outputs = {}
                        def capture(family):
                            def hook(module, args, output):
                                value = output[0] if isinstance(output, tuple) else output
                                outputs[family] = value.detach().clone()[:, support['rows']]
                            return hook
                        with ExitStack() as stack:
                            for family in ('FA', 'D'):
                                stack.callback(native.sites[a[family+'_layer']].register_forward_hook(capture(family)).remove)
                            disabled = ('FA' if repeat % 2 == 0 else 'D') if mode == 'disabled' else None
                            if disabled:
                                source = torch.zeros((1,len(item.input_ids)), dtype=torch.bool, device=next(model.parameters()).device)
                                rows = source.clone(); rows[:,support['rows']] = True
                                stack.enter_context(native_site(native.sites[a[disabled+'_layer']], arm='H-T', source=source, rows=rows, enabled=False))
                            logits, ids = native(item.task,item)
                        if list(ids) != retained['answer_token_ids'] or any(not torch.isfinite(v).all() for v in outputs.values()):
                            raise ValueError('nonfinite repeat or answer IDs mismatch')
                        if reference is None:
                            reference = outputs
                        result['repeats'].append(dict(example_id=key, mode=mode, repeat=repeat, disabled_family=disabled,
                            answer=_answer(logits,ids,clean), displacement={f:_r(reference[f],outputs[f]) for f in ('FA','D')}))
        result['finite_path'] = _finite(result)
        if not result['finite_path']:
            raise ValueError('nonfinite checkpoint evidence')
    finally:
        for module, mode in modes:
            module.training = mode
    return result


def _validate_answer(answer):
    for field in ('conditional_log_probabilities', 'native_answer_log_probabilities', 'answer_logits'):
        values = answer[field]
        if len(values) != 3 or not _finite(values):
            raise ValueError('complete finite three-answer evidence required')
    logs = answer['conditional_log_probabilities']
    native = answer['native_answer_log_probabilities']
    if any(x > 1e-6 for x in logs+native) or abs(sum(math.exp(x) for x in logs)-1) > 1e-5:
        raise ValueError('invalid answer log probabilities')
    logits = answer['answer_logits']
    logits_z = max(logits)+math.log(sum(math.exp(x-max(logits)) for x in logits))
    if any(abs(x-logits_z-y)>1e-5 for x,y in zip(logits,logs)):
        raise ValueError('inconsistent answer logits')
    z = max(native)+math.log(sum(math.exp(x-max(native)) for x in native))
    if z > 1e-6 or any(abs(x-z-y)>1e-5 for x,y in zip(native,logs)):
        raise ValueError('inconsistent native/conditional probabilities')


def _effect(clean, changed, q):
    _validate_answer(clean); _validate_answer(changed)
    kl = max(0.,sum(math.exp(p)*(p-r) for p,r in zip(clean['conditional_log_probabilities'],changed['conditional_log_probabilities'])))
    ce = sum(w*(p-r) for w,p,r in zip(q,clean['native_answer_log_probabilities'],changed['native_answer_log_probabilities']))
    return kl, ce


def checkpoint_summary(evidence, *, expected_identity, votes, operator_locked):
    """Join vote-q only after operator lock, outside all native execution."""
    if operator_locked is not True or evidence['identity'] != expected_identity or evidence['finite_path'] is not True or not _finite(evidence):
        raise ValueError('locked operator and exact finite checkpoint identity required')
    schedule = expected_identity['assignments']
    if evidence['arm'] not in ARMS or len(schedule) != 32 or [r['assignment'] for r in evidence['rows']] != schedule:
        raise ValueError('fixed first32 coverage/order required')
    expected_repeats = [(a['example_id'],m,r) for a in schedule[:4] for m in ('untouched','disabled') for r in range(10)]
    if [(r['example_id'],r['mode'],r['repeat']) for r in evidence['repeats']] != expected_repeats:
        raise ValueError('exact80 repeat fixtures required')
    clean = {r['example_id']:r['clean'] for r in evidence['rows']}
    for key in clean:
        q = votes[key]
        if len(q)!=3 or not _finite(q) or any(x<0 for x in q) or abs(sum(q)-1)>1e-6:
            raise ValueError('finite normalized vote-q required')
    floors = []
    for r in evidence['repeats']:
        expected_disabled = ('FA' if r['repeat']%2==0 else 'D') if r['mode']=='disabled' else None
        if r['disabled_family'] != expected_disabled or set(r['displacement']) != {'FA','D'} or any(x<0 for x in r['displacement'].values()):
            raise ValueError('repeat disabled-family/displacement mismatch')
        kl, ce = _effect(clean[r['example_id']],r['answer'],votes[r['example_id']])
        floors.append((kl,abs(ce)))
    measured = []
    for row in evidence['rows']:
        if row['example_id'] != row['assignment']['example_id'] or set(row['probes']) != {'FA','D'}:
            raise ValueError('row/family mismatch')
        for family, probe in row['probes'].items():
            if probe['endpoint']['family']!=family or probe['endpoint']['site']!=row['assignment'][family+'_layer']:
                raise ValueError('site mismatch')
            kl,ce = _effect(row['clean'],probe['answer'],votes[row['example_id']])
            measured.append(dict(family=family,block=row['assignment']['block'],conditional_kl=kl,ce_difference=ce,abs_ce_difference=abs(ce)))
    families = {}
    for family in ('FA','D'):
        values = [x for x in measured if x['family']==family]
        families[family] = dict(count=len(values),mean_kl=sum(x['conditional_kl'] for x in values)/32,
            mean_abs_ce_difference=sum(x['abs_ce_difference'] for x in values)/32,
            by_block=dict(Counter(x['block'] for x in values)))
    return dict(identity=expected_identity, raw_evidence_sha256=digest(evidence), votes_sha256=digest({k:votes[k] for k in clean}), arm=evidence['arm'], families=families, kl_floor=max(x[0] for x in floors),
                ce_floor=max(x[1] for x in floors), mean_kl=sum(x['conditional_kl'] for x in measured)/64,
                mean_abs_ce_difference=sum(x['abs_ce_difference'] for x in measured)/64,
                measurements=measured, block_counts=dict(Counter(x['block'] for x in measured)),
                block4=[x for x in measured if x['block']==4])


def observational_report(updates, checkpoints, *, expected_checkpoints, checkpoint_reduction=None):
    """Unavailable evidence never becomes an inert-arm claim.

    checkpoints: ordered {update, summary}; expected_checkpoints binds identities.
    updates: {update, window_size, exposure}, including every unsupported row.
    """
    report = dict(status='UNAVAILABLE', observationally_inert=None, reasons=[])
    try:
        if checkpoint_reduction != 'per_family_mean32':
            raise ValueError('unfrozen checkpoint reduction: explicitly freeze per_family_mean32 before H run')
        if not checkpoints or len(checkpoints)!=len(expected_checkpoints) or not _finite((updates,checkpoints)):
            raise ValueError('complete finite checkpoint coverage required')
        positions = [c['update'] for c in checkpoints]
        if positions != sorted(set(positions)) or [dict(update=c['update'], identity=c['summary']['identity'], summary_sha256=digest(c['summary'])) for c in checkpoints] != expected_checkpoints:
            raise ValueError('checkpoint order/identity mismatch')
        if any(type(p) is not int or p<0 for p in positions) or positions[0]!=0 or positions[-1]!=len(updates):
            raise ValueError('initial and final completed-update checkpoints required')
        identities = [c['summary']['identity'] for c in checkpoints]
        lineage = [{k:v for k,v in ident.items() if k != 'checkpoint_sha256'} for ident in identities]
        if any(value != lineage[0] for value in lineage):
            raise ValueError('mixed checkpoint seed/operator/input/cache lineage')
        arms = {c['summary']['arm'] for c in checkpoints}
        if len(arms)!=1 or not arms.issubset(ARMS):
            raise ValueError('one supported arm required')
        for c in checkpoints:
            s=c['summary']
            if any(s[k]<0 for k in ('kl_floor','ce_floor','mean_kl','mean_abs_ce_difference')):
                raise ValueError('negative checkpoint metric')
        trajectory=[]; ordinals=[]
        if [u['update'] for u in updates] != list(range(len(updates))):
            raise ValueError('complete consecutive optimizer updates required')
        for u in updates:
            rows=u['exposure']; w=u['window_size']
            if type(w) is not int or w<=0 or len(rows)!=w:
                raise ValueError('actual full optimizer window_size required')
            left=[c for c in checkpoints if c['update']<=u['update']]
            right=[c for c in checkpoints if c['update']>=u['update']+1]
            if not left or not right:
                raise ValueError('missing initial/final bracketing checkpoint floors')
            kfloor=max(left[-1]['summary']['kl_floor'],right[0]['summary']['kl_floor'])
            cfloor=max(left[-1]['summary']['ce_floor'],right[0]['summary']['ce_floor'])
            for r in rows:
                ordinals.append(r['ordinal'])
                if r['family'] not in ('FA','D') or not 0<=r['layer']<32 or (r['family']=='FA') != (r['layer']%4==3):
                    raise ValueError('training family/layer mismatch')
                if type(r['supported']) is not bool or type(r['duplicate_skipped']) is not bool:
                    raise ValueError('explicit support/duplicate evidence required')
                if r['conditional_kl'] < -1e-6 or r['abs_ce_difference']<0 or abs(abs(r['ce_difference'])-r['abs_ce_difference'])>1e-6:
                    raise ValueError('invalid exposure metrics')
                if not r['supported'] and (not r['duplicate_skipped'] or any(r[k]!=0 for k in ('conditional_kl','ce_difference','abs_ce_difference'))):
                    raise ValueError('unsupported exposure must be duplicate-skipped zero')
            supported=any(r['supported'] for r in rows)
            kl=sum(max(0.,r['conditional_kl']) for r in rows)/w
            ce=sum(r['abs_ce_difference'] for r in rows)/w
            trajectory.append(dict(update=u['update'],window_size=w,supported=supported,mean_kl=kl,
                mean_abs_ce_difference=ce,kl_floor=kfloor,ce_floor=cfloor,within_floors=kl<=kfloor and ce<=cfloor,
                family_counts=dict(Counter(r['family'] for r in rows)),block4=[r for r in rows if r['layer']//4==4]))
        if ordinals != list(range(len(ordinals))):
            raise ValueError('complete ordered exposure ordinals required')
        supported=[u for u in trajectory if u['supported']]
        if not supported:
            raise ValueError('no supported optimizer updates')
        within=sum(u['within_floors'] for u in supported)
        for c in checkpoints:
            s=c['summary']
            if set(s['families']) != {'FA','D'} or not _sha(s['raw_evidence_sha256']) or not _sha(s['votes_sha256']):
                raise ValueError('complete checkpoint family/evidence/vote identity required')
            for values in s['families'].values():
                if values['count']!=32 or values['mean_kl']<0 or values['mean_abs_ce_difference']<0:
                    raise ValueError('complete family32 checkpoint means required')
        dev_ok=all(f['mean_kl']<=c['summary']['kl_floor'] and f['mean_abs_ce_difference']<=c['summary']['ce_floor']
                   for c in checkpoints for f in c['summary']['families'].values())
        inert=5*within>=4*len(supported) and dev_ok
        report.update(status='AVAILABLE', observationally_inert=inert, supported_updates=len(supported),
            updates_within_floors=within, fraction_within_floors=within/len(supported), checkpoints_within_floors=dev_ok,
            trajectory=trajectory, checkpoints=checkpoints, checkpoint_reduction=checkpoint_reduction,
            interpretation='observationally inert at logged prediction/loss resolution' if inert else 'observed manipulation exceeds frozen flag criteria')
    except (KeyError,TypeError,ValueError,IndexError,OverflowError) as exc:
        report['reasons']=[str(exc)]
    return report
