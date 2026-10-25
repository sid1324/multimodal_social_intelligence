"""Explicit prospective training dose; diagnostic evidence grants no job authority."""
from dataclasses import dataclass
from math import exp, log, isfinite
from statistics import median
from ..hashing import sha256_file
from ..io_utils import read_json
from .cache import digest
from .diagnostic_potency import AMENDMENT, COHORT_SHA256, ORIGINAL_SHA256

PROTOCOL = 'prospective-srer-training-eta050-v1'
_VALIDATED = object()
APPROVAL_SHA256 = '53cedd4cbc4f61e3f299d76473108c5d1203196ba54e26d96bfa61782da6e645'
DIAGNOSTIC_SHA256 = 'f73b0eb924302cef9e8d9964dbe6bde7508e612fb43c9d0ad69864b2bb3d7079'


@dataclass(frozen=True)
class Eta050Protocol:
    """Construct through validate_protocol; never substitutes for Seal B."""
    identity: tuple
    _validation: object = None

    def __post_init__(self):
        if self._validation is not _VALIDATED:
            raise ValueError('construct eta050 protocol through artifact validation')

    def as_dict(self):
        return dict(self.identity)


def require_dose(eta, protocol=None):
    if eta in (0, .25):
        if protocol is not None:
            raise ValueError('eta050 protocol cannot be used with the original dose')
        return
    if eta != .5 or not isinstance(protocol, Eta050Protocol):
        raise ValueError('eta=.5 requires validated prospective training protocol')
    identity = protocol.as_dict()
    if identity.get('protocol') != PROTOCOL or identity.get('diagnostic_sha256') != DIAGNOSTIC_SHA256:
        raise ValueError('invalid prospective protocol identity')


def _artifact(spec, expected=None):
    actual = sha256_file(spec['path'])
    if actual != spec['sha256'] or (expected is not None and actual != expected):
        raise ValueError('eta050 artifact bytes mismatch')
    return read_json(spec['path'])


def validate_diagnostic(data, original):
    """Check raw finite evidence and cohort, never the stored PASS flag."""
    if (data.get('amendment') != AMENDMENT or data.get('dose') != .5 or
        data.get('evidence_kind') != 'prospective_frozen_diagnostic' or
        data.get('original_potency_sha256') != ORIGINAL_SHA256 or
        data.get('cohort_sha256') != COHORT_SHA256 or
        data.get('intervention_training_allowed') is not False or
        data.get('seal_b_authorization') is not False):
        raise ValueError('wrong prospective diagnostic identity/authority')
    for key in ('assignments', 'pins', 'eligible_population_sha256', 'cache_index_sha256'):
        if data[key] != original[key]:
            raise ValueError('diagnostic original cohort/pins/cache mismatch')
    assignments = data['assignments']
    if len(assignments) != 128 or digest(assignments) != COHORT_SHA256:
        raise ValueError('fixed128 assignment cohort required')
    expected = {(a['example_id'], site) for a in assignments[:4] for site in range(32)}
    parity = data['parity']
    if len(parity) != 128 or {(r['example_id'], r['site']) for r in parity} != expected:
        raise ValueError('all128 unique active clean parity entries required')
    for row in parity:
        d = row['differences']
        if (row['custom_calls'] != 1 or row['argmax_same'] is not True or
            not all(isfinite(v) and v >= 0 for v in d.values()) or
            d['logprobs'] > 1e-6 or d['relative_site'] > 1e-5):
            raise ValueError('active clean parity criteria failed')
    repeats = data['repeats']
    if len(repeats) != 4 or [r['example_id'] for r in repeats] != [a['example_id'] for a in assignments[:4]]:
        raise ValueError('fixed four repeat groups required')
    for r in repeats:
        floors = r['floors']
        values = [floors[k] for k in ('kl', 'logits', 'logprobs')] + list(floors['sites'].values())
        if r['repeats'] != 10 or set(floors['sites']) != {str(i) for i in range(32)} or any(v != 0 for v in values):
            raise ValueError('exact ten-repeat numerical floors required')
    if len(data['rows']) != 128 or len(original['rows']) != 128:
        raise ValueError('fixed128 dose records required')
    summaries = {family: ([], []) for family in ('D', 'FA')}
    for a, row, old in zip(assignments, data['rows'], original['rows']):
        if (row['assignment'] != a or row['example_id'] != a['example_id'] or
            row['source_cache_sha256'] != old['source_cache_sha256'] or
            row['grid']['clean'] != old['grid']['clean']):
            raise ValueError('dose rows/clean reference/source drift')
        if set(row['grid']) != {'clean', 'D_target', 'D_random', 'FA_target', 'FA_random'}:
            raise ValueError('complete prescribed dose grid required')
        def logp(values):
            if len(values) != 3 or not all(isfinite(v) for v in values):
                raise ValueError('finite three-answer logits required')
            m = max(values)
            z = m + log(sum(exp(v-m) for v in values))
            return [v-z for v in values]
        p = logp(row['grid']['clean']['answer_logits'])
        for family in ('D', 'FA'):
            for support in ('target', 'random'):
                key = family + '_' + support
                cell = row['grid'][key]
                if cell.get('eta') != .5:
                    raise ValueError('every prospective dose cell must use eta=.5')
                endpoint = cell['endpoint']
                old_endpoint = old['grid'][key]['endpoint']
                if (endpoint['mask_sha256'] != digest(dict(source=old_endpoint['source'], rows=old_endpoint['rows'])) or
                    endpoint['site'] != a[family+'_layer'] or endpoint['family'] != family):
                    raise ValueError('dose mask/site identity mismatch')
                q = logp(cell['answer_logits'])
                kl = sum(exp(x)*(x-y) for x,y in zip(p,q))
                displacement = endpoint['displacement']
                if (not isfinite(displacement) or displacement < 0 or
                    not isfinite(cell['conditional_kl_float64']) or
                    not isfinite(cell['conditional_kl']) or
                    abs(kl-cell['conditional_kl_float64']) > 1e-12 or
                    abs(kl-cell['conditional_kl']) > 2e-7):
                    raise ValueError('dose finite displacement/KL recomputation failed')
                if support == 'target':
                    summaries[family][0].append(kl)
                    summaries[family][1].append(displacement)
    if any(median(kl) < 1e-4 or median(r) <= 1e-5 for kl,r in summaries.values()):
        raise ValueError('both targeted families must meet fixed potency criteria')


def validate_protocol(spec):
    """Return a dose capability after verifying actual immutable artifacts.

    The spec is independently frozen in profile/Seal B. Acquisition identities
    are validated by the ordinary cache readers against the current serializers.
    """
    if spec.get('protocol') != PROTOCOL or spec.get('eta') != .5:
        raise ValueError('explicit eta050 training amendment required')
    diagnostic = _artifact(spec['diagnostic'], DIAGNOSTIC_SHA256)
    original = _artifact(spec['original_potency'], ORIGINAL_SHA256)
    # Approval is prose; verify bytes without interpreting it as a job authority.
    approval = spec['user_approval']
    if sha256_file(approval['path']) != approval['sha256'] or approval['sha256'] != APPROVAL_SHA256:
        raise ValueError('user amendment approval bytes mismatch')
    validate_diagnostic(diagnostic, original)
    return Eta050Protocol(tuple(sorted(dict(protocol=PROTOCOL, eta=.5,
        diagnostic_sha256=DIAGNOSTIC_SHA256, original_potency_sha256=ORIGINAL_SHA256,
        user_approval_sha256=approval['sha256'], amendment_sha256=digest(spec)).items())), _VALIDATED)


def require_arm(arm, protocol):
    if protocol is None:
        return
    from .family_controls import is_family, ARMS
    if is_family(protocol):
        if arm not in ARMS:
            raise ValueError('family-control amendment authorizes F-T/D-T only')
    elif arm not in ('H-T','H-R'):
        raise ValueError('eta050 amendment authorizes H-T/H-R only')


def launch_protocol(launch):
    """Explicit opt-in shared by profile, Seal B, trainer and C2 consumers."""
    spec = launch.get('training_amendment')
    family = launch.get('family_control_amendment')
    if spec is None:
        if family is not None:
            raise ValueError('family controls require the original eta050 parent amendment')
        return None
    protocol = validate_protocol(spec)
    if family is not None:
        from .family_controls import validate_family_protocol
        return validate_family_protocol(protocol, family)
    return protocol
