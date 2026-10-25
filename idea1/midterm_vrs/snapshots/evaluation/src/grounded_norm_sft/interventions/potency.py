"""Frozen, input-only 9.C potency evidence. No loader, optimizer, or label input.

The aggregator is deliberately independent of the native runner: absent evidence is
INCOMPLETE, small cohorts are DEFER, and only complete finite evidence can PASS.
"""
from collections import Counter
from contextlib import ExitStack
from statistics import median
import math

import torch

from ..constants import POTENCY_SORT_PREFIX
from ..hashing import sort_key_sha256
from ..schedules import potency_schedule
from .cache import digest
from .forward import NativeForward, validated_supports
from .native_adapters import native_site, diagnostic_eta_one

FAMILIES = ('FA', 'D')
GRID = ('clean', 'shuffle', 'FA_target', 'FA_random', 'FA_endpoint',
        'D_target', 'D_random', 'D_endpoint')


def assignments(eligible_ids):
    ids = list(eligible_ids)
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('unique nonempty eligible dev IDs required')
    cohort = sorted(ids, key=lambda x: sort_key_sha256(POTENCY_SORT_PREFIX, x))[:128]
    return [dict(example_id=key, block=b, delta_offset=d, FA_layer=4*b+3, D_layer=4*b+d)
            for key, (b, d) in zip(cohort, potency_schedule(len(cohort)))]


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(x) for x in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def aggregate(evidence):
    """Validate exact coverage and predeclared thresholds; never infer missing zeros."""
    report = dict(status='UNRUN', intervention_training_allowed=False,
                  branch='B0/B1 only', families={}, reasons=[])
    if evidence is None:
        return report
    report['evidence_kind'] = evidence.get('evidence_kind', 'unspecified')
    try:
        expected = assignments(evidence['eligible_dev_ids'])
        if evidence['assignments'] != expected:
            raise ValueError('cohort or shared site assignments mismatch')
        n = len(expected)
        report['cohort_size'] = n
        if n < 64:
            report.update(status='DEFER', reasons=['fewer than 64 eligible dev inputs'])
            return report
        if not _finite(evidence) or evidence.get('finite_path') is not True:
            raise ValueError('nonfinite or unverified finite path')
        if any(not _sha(evidence.get('pins', {}).get(k)) for k in
               ('model_sha256', 'processor_sha256', 'template_sha256')) or not _sha(evidence.get('shuffle_map_sha256')):
            raise ValueError('missing immutable model/processor/template or shuffle identities')
        rows = evidence['rows']
        if len(rows) != n or [r['example_id'] for r in rows] != [a['example_id'] for a in expected]:
            raise ValueError('missing, duplicated, or reordered input evidence')
        floors = evidence['floors']
        expected_floors = {(a['example_id'], mode, repeat) for a in expected[:4]
                           for mode in ('untouched', 'disabled') for repeat in range(10)}
        actual_floors = [(f['example_id'], f['mode'], f['repeat']) for f in floors]
        if len(actual_floors) != 80 or set(actual_floors) != expected_floors:
            raise ValueError('floor fixtures must be exactly first four by 10 untouched and 10 disabled')
        for row, assignment in zip(rows, expected):
            if row['assignment'] != assignment or not _sha(row.get('source_cache_sha256')):
                raise ValueError('missing source identity or changed assignment')
            if set(row['grid']) != set(GRID):
                raise ValueError('incomplete eight-forward grid')
            for probe in row['grid'].values():
                if len(probe['conditional_probabilities']) != 3 or len(probe['native_answer_probabilities']) != 3:
                    raise ValueError('three answer probabilities required')
                conditional = probe['conditional_probabilities']
                full = probe['native_answer_probabilities']
                if any(x < 0 or x > 1 for x in conditional + full) or abs(sum(conditional)-1) > 1e-5 or abs(sum(full)-probe['valid_answer_mass']) > 1e-5:
                    raise ValueError('inconsistent answer probabilities')
                if not 0 <= probe['valid_answer_mass'] <= 1.000001 or probe['conditional_kl'] < 0:
                    raise ValueError('invalid probability or negative KL')
            clean = row['grid']['clean']
            for probe in row['grid'].values():
                for field in ('conditional_log_probabilities', 'native_answer_log_probabilities', 'answer_logits', 'answer_logit_changes'):
                    if len(probe[field]) != 3:
                        raise ValueError('incomplete answer vector')
                logs = probe['conditional_log_probabilities']
                native_logs = probe['native_answer_log_probabilities']
                if any(x > 1e-6 for x in logs + native_logs):
                    raise ValueError('positive answer log probability')
                if any(abs(math.exp(x)-y) > 1e-6 for x,y in zip(logs,probe['conditional_probabilities'])) or any(abs(math.exp(x)-y) > 1e-6 for x,y in zip(native_logs,probe['native_answer_probabilities'])):
                    raise ValueError('inconsistent answer log probabilities')
                logits = probe['answer_logits']
                z = max(logits) + math.log(sum(math.exp(x-max(logits)) for x in logits))
                native_z = max(native_logs) + math.log(sum(math.exp(x-max(native_logs)) for x in native_logs))
                if any(abs((x-z)-lp) > 1e-5 for x,lp in zip(logits,logs)) or any(abs((x-native_z)-lp) > 1e-5 for x,lp in zip(native_logs,logs)):
                    raise ValueError('answer logits and conditional/native probabilities disagree')
                kl = max(0., sum(p*(a-b) for p,a,b in zip(clean['conditional_probabilities'], clean['conditional_log_probabilities'], logs)))
                argmax = max(range(3), key=lambda j: logs[j])
                if abs(kl-probe['conditional_kl']) > 1e-6 or probe['argmax'] != argmax or probe['argmax_flip'] != (argmax != clean['argmax']):
                    raise ValueError('inconsistent derived KL or argmax')
                if any(abs(d-(b-a)) > 1e-5 for d,a,b in zip(probe['answer_logit_changes'],clean['answer_logits'],probe['answer_logits'])):
                    raise ValueError('inconsistent answer-logit changes')
            for family in FAMILIES:
                for condition in ('target', 'random', 'endpoint'):
                    p = row['grid'][family+'_'+condition]
                    if p.get('eta') != (1. if condition == 'endpoint' else .25):
                        raise ValueError('probe dose mismatch')
                    if p['endpoint']['family'] != family or p['endpoint']['site'] != assignment[family+'_layer']:
                        raise ValueError('endpoint family/site mismatch')
                    endpoint = p['endpoint']
                    for field in ('readout_norm_ratio', 'excluded_rms', 'postgate_rms', 'projected_norm'):
                        if field not in endpoint:
                            raise ValueError('missing native endpoint diagnostics')
                    if family == 'D' and any(field not in endpoint for field in
                        ('normalized_rms', 'native_epsilon', 'low_ratio', 'low_rms', 'joint_low_signal')):
                        raise ValueError('missing Delta low-signal diagnostics')
                    if p['endpoint']['displacement'] < 0:
                        raise ValueError('negative displacement')
        for fixture in floors:
            if fixture['conditional_kl'] < 0 or fixture['max_abs_vote_ce_difference_bound'] < 0 or any(fixture['families'][f]['displacement'] < 0 for f in FAMILIES):
                raise ValueError('negative numerical floor')
        for family in FAMILIES:
            floor_r = max(f['families'][family]['displacement'] for f in floors)
            floor_kl = max(max(0., f['conditional_kl']) for f in floors)
            if floor_r < 0:
                raise ValueError('negative numerical floor')
            r = [row['grid'][family+'_target']['endpoint']['displacement'] for row in rows]
            kl = [row['grid'][family+'_target']['conditional_kl'] for row in rows]
            r_threshold, kl_threshold = max(1e-5, 100*floor_r), max(1e-4, 100*floor_kl)
            passed = median(r) > r_threshold and median(kl) >= kl_threshold
            report['families'][family] = dict(passed=passed, median_r=median(r), median_kl=median(kl),
                r_floor=floor_r, kl_floor=floor_kl, r_threshold=r_threshold, kl_threshold=kl_threshold,
                by_block=dict(Counter(a['block'] for a in expected)),
                by_site=dict(Counter(a[family+'_layer'] for a in expected)),
                displacement=r, conditional_kl=kl,
                random_kl=[row['grid'][family+'_random']['conditional_kl'] for row in rows],
                endpoint_kl=[row['grid'][family+'_endpoint']['conditional_kl'] for row in rows],
                argmax_flips=sum(row['grid'][family+'_target']['argmax_flip'] for row in rows),
                inactive_site_count=sum(x <= r_threshold for x in r))
        report['max_abs_vote_ce_difference_bound'] = max(f['max_abs_vote_ce_difference_bound'] for f in floors)
        passed = all(f['passed'] for f in report['families'].values())
        report['seal_b_authorization'] = False
        report['engineering_gate_passed'] = passed
        report.update(status='PASS' if passed else 'FAIL', intervention_training_allowed=False,
                      branch='intervention engineering gate passed' if passed else 'B0/B1 only',
                      shuffle_kl=[r['grid']['shuffle']['conditional_kl'] for r in rows])
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        report.update(status='INCOMPLETE', reasons=[str(exc)])
    return report


def _answer(logits, ids, clean=None):
    logits = logits.detach().float()
    if logits.ndim != 1 or len(ids) != 3 or len(set(ids)) != 3 or not torch.isfinite(logits).all():
        raise ValueError('finite native final-row logits and three unique answer token IDs required')
    ids = list(ids)
    logp = logits[ids].log_softmax(-1)
    full = logits.log_softmax(-1)[ids].exp()
    baseline = logp if clean is None else torch.tensor(clean['conditional_log_probabilities'], device=logits.device)
    kl = (baseline.exp() * (baseline-logp)).sum().clamp_min(0).item()
    return dict(native_answer_probabilities=full.tolist(), native_answer_log_probabilities=logits.log_softmax(-1)[ids].tolist(), conditional_probabilities=logp.exp().tolist(),
                conditional_log_probabilities=logp.tolist(), valid_answer_mass=full.sum().item(),
                conditional_kl=kl, argmax=int(logp.argmax()),
                argmax_flip=bool(logp.argmax().item() != baseline.argmax().item()),
                answer_logits=logits[ids].tolist(),
                answer_logit_changes=[0.]*3 if clean is None else (logits[ids]-torch.tensor(clean['answer_logits'], device=logits.device)).tolist())


def _r(clean, changed):
    return float((changed.float()-clean.float()).norm() / clean.float().norm().clamp_min(1e-8))


def run(model, serialized, shuffled, *, cache_paths, pins, image_sha256, alignment_ids,
        shuffle_map, shuffle_map_sha256):
    """Run supplied frozen CPU base, no labels; return durable-ready detached evidence.

    ``shuffled`` must be serialized with Appendix A's frozen dev mapping. Its map
    digest is checked here; the serializer remains responsible for donor-image
    construction. Eta=1 uses the separate diagnostic-only excluded endpoint scope.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('supplied frozen CPU or native device base required')
    if any('lora' in name.lower() or 'adapter' in name.lower() for name, _ in model.named_modules()):
        raise ValueError('frozen base must contain no adapters')
    if digest(shuffle_map) != shuffle_map_sha256:
        raise ValueError('shuffle mapping identity mismatch')
    supports = validated_supports(serialized, cache_paths, pins=pins, image_sha256=image_sha256,
                                  alignment_ids=alignment_ids)
    eligible = [key for key, rec in supports.items() if rec['payload']['support']['supported']]
    schedule = assignments(eligible)
    evidence = dict(evidence_kind='supplied_frozen_base', eligible_dev_ids=eligible,
                    assignments=schedule, pins=dict(pins), shuffle_map_sha256=shuffle_map_sha256,
                    rows=[], floors=[], finite_path=False)
    if len(schedule) < 64:
        evidence['report'] = aggregate(evidence)
        return evidence
    for a in schedule:
        key = a['example_id']
        if key not in shuffled or key not in shuffle_map:
            raise ValueError('fixed dev shuffle coverage missing')
        for item in (serialized[key], shuffled[key]):
            if any(k in item.model_inputs for k in ('labels', 'label')):
                raise ValueError('labels cannot be model inputs')
    native = NativeForward(model, alignment_ids=alignment_ids)
    native.bind(serialized)
    native.bind(shuffled)
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for ordinal, a in enumerate(schedule):
                key = a['example_id']; item = serialized[key]
                support = supports[key]['payload']['support']
                outputs = {}
                def capture(family):
                    def hook(module, args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        outputs[family] = value.detach().clone()[:, support['rows']]
                        if not torch.isfinite(outputs[family]).all():
                            raise FloatingPointError('nonfinite native block output')
                    return hook
                with ExitStack() as stack:
                    for family in FAMILIES:
                        stack.callback(native.sites[a[family+'_layer']].register_forward_hook(capture(family)).remove)
                    logits, ids = native(item.task, item)
                clean_outputs = dict(outputs)
                clean = _answer(logits, ids)
                grid = {'clean': clean}
                changed, changed_ids = native(shuffled[key].task, shuffled[key])
                if list(changed_ids) != list(ids):
                    raise ValueError('shuffle answer IDs mismatch')
                grid['shuffle'] = _answer(changed, ids, clean)
                for family in FAMILIES:
                    row = dict(site_layer=a[family+'_layer'])
                    for condition in ('target', 'random', 'endpoint'):
                        diagnostics = []
                        native.diagnostic = diagnostics.append
                        if condition == 'endpoint':
                            source = torch.zeros((1, len(item.input_ids)), dtype=torch.bool, device=next(model.parameters()).device)
                            rows = torch.zeros_like(source)
                            source[0, [support['patches'][i] for i in support['targeted']]] = True
                            rows[0, support['rows']] = True
                            context = diagnostic_eta_one(native.sites[row['site_layer']], source=source, rows=rows,
                                                         diagnostic=diagnostics.append, ordinal=ordinal)
                        else:
                            context = native.perturbed(item, support, row,
                                arm='H-R' if condition == 'random' else 'H-T', eta=.25)
                        with context:
                            changed, changed_ids = native(item.task, item)
                        if list(changed_ids) != list(ids) or len(diagnostics) != 1:
                            raise ValueError('exactly one matching native endpoint record required')
                        probe = _answer(changed, ids, clean)
                        probe['endpoint'] = diagnostics[0]
                        probe['eta'] = 1. if condition == 'endpoint' else .25
                        grid[family+'_'+condition] = probe
                evidence['rows'].append(dict(example_id=key, assignment=a, grid=grid,
                    source_cache_sha256=supports[key]['sha256']))
                if ordinal < 4:
                    for mode in ('untouched', 'disabled'):
                        for repeat in range(10):
                            outputs.clear()
                            with ExitStack() as stack:
                                for family in FAMILIES:
                                    stack.callback(native.sites[a[family+'_layer']].register_forward_hook(capture(family)).remove)
                                if mode == 'disabled':
                                    source = torch.zeros((1, len(item.input_ids)), dtype=torch.bool, device=next(model.parameters()).device)
                                    rows = source.clone(); rows[:, support['rows']] = True
                                    stack.enter_context(native_site(native.sites[a['FA_layer' if repeat % 2 == 0 else 'D_layer']], arm='H-T',
                                        source=source, rows=rows, enabled=False))
                                changed, changed_ids = native(item.task, item)
                            probe = _answer(changed, changed_ids, clean)
                            ce = max(abs(x-y) for x,y in zip(clean['native_answer_log_probabilities'],
                                                            probe['native_answer_log_probabilities']))
                            evidence['floors'].append(dict(example_id=key, mode=mode, repeat=repeat,
                                conditional_kl=probe['conditional_kl'], clean_native_answer_log_probabilities=clean['native_answer_log_probabilities'], native_answer_log_probabilities=probe['native_answer_log_probabilities'], disabled_family=(('FA' if repeat % 2 == 0 else 'D') if mode == 'disabled' else None), max_abs_vote_ce_difference_bound=ce,
                                families={f:dict(displacement=_r(clean_outputs[f], outputs[f])) for f in FAMILIES}))
        evidence['finite_path'] = True
    finally:
        for module, mode in modes:
            module.training = mode
    evidence['report'] = aggregate(evidence)
    return evidence
