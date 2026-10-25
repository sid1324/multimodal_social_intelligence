"""Bounded hybrid orchestration; clean dev forwards also supply C2 and gate health.

This module supplies no authorization. Production callers must pass the launch
checks; fixture callers use random CPU models and never acquire GPU authority.
"""
from contextlib import ExitStack
from pathlib import Path
import torch
from ..io_utils import read_json, write_once_json
from ..runtime.model import set_train_mode, frozen_state_sha256
from ..runtime.train import Budget as RunBudget, TrainingRun
from .cache import digest
from .forward import NativeForward
from .data import SupportInputs
from . import manipulation

REDUCTION = 'per_family_mean32'


def put(path, value):
    """Crash-repairable immutable evidence, including a missing sidecar."""
    from ..runtime.evaluate import _write_or_verify, _json_bytes
    _write_or_verify(Path(path), _json_bytes(value))
    return value


def _stats(value, beta=False):
    value = value.detach().float().flatten()
    if not value.numel() or not torch.isfinite(value).all():
        raise FloatingPointError('empty/nonfinite native gate health')
    result = dict(count=value.numel(), mean=float(value.mean()),
                  quantiles=torch.quantile(value, torch.tensor([.01, .5, .99], device=value.device)).cpu().tolist())
    if beta:
        result.update(fraction_beta_lt_1e_minus3=float((value < 1e-3).float().mean()),
                      fraction_beta_gt_1minus1e_minus3=float((value > 1-1e-3).float().mean()))
    return result


def gate_health(stack, native, item, support):
    """Observe native projections on the existing clean forward; no extra projection."""
    captured = {}
    segments = dict(image=support['patches'], post_action_suffix=support['rows'])
    for layer, module in native.sites.items():
        if layer % 4 == 3:
            continue
        captured[layer] = {}
        def capture(field, layer=layer, module=module):
            def hook(unused_module, args, output):
                with torch.no_grad():
                    value = (output.sigmoid() if field == 'beta' else
                             -module.A_log.float().exp() * torch.nn.functional.softplus(output.float() + module.dt_bias))
                    captured[layer][field] = {name: _stats(value[:, indices], field == 'beta')
                                              for name, indices in segments.items()}
            return hook
        stack.callback(module.in_proj_a.register_forward_hook(capture('g')).remove)
        stack.callback(module.in_proj_b.register_forward_hook(capture('beta')).remove)
    return captured


class CheckpointDev:
    """Per-input durable clean dev; pending C2 resumes without clean re-forwards."""
    def __init__(self, model, inputs, cache_paths, assignments, directory, *, pins,
                 alignment_ids, seed, operator_sha256, arm, budget, dose_protocol=None):
        self.dose_protocol = dose_protocol
        self.model, self.inputs, self.paths = model, inputs, cache_paths
        self.assignments, self.root = assignments, Path(directory)
        self.pins, self.alignment = pins, alignment_ids
        self.seed, self.operator, self.arm, self.budget = seed, operator_sha256, arm, budget
        self.supports = SupportInputs(inputs, cache_paths, pins=pins, image_sha256=inputs.image_sha256,
                                      alignment_ids=alignment_ids)
        # run() independently verifies cohort eligibility against all cached records.
        if not 64 <= len(assignments) <= 128:
            raise ValueError('C2 requires the original eligible potency cohort (64..128)')
        from .potency import assignments as make_assignments
        eligible = [key for key in inputs if self.supports[key]['payload']['support']['supported']]
        if assignments != make_assignments(eligible):
            raise ValueError('original frozen potency cohort/support membership mismatch')
        self.native = NativeForward(model, alignment_ids=alignment_ids)
        self.native.bind(inputs)

    def __call__(self, epoch, checkpoint):
        checksum = checkpoint.get('manifest_sha256') or checkpoint['adapter_sha256']
        root = self.root / ('u%05d' % checkpoint['next_update'])
        ident = manipulation.identity(checkpoint_sha256=checksum, seed=self.seed,
            operator_sha256=self.operator, serialized=self.inputs, records=self.supports,
            schedule=self.assignments[:32], dose_protocol=self.dose_protocol)
        scientific = dict(identity=ident, all_dev_ids_sha256=digest(list(self.inputs)), arm=self.arm,
                          dev_source=self.inputs.identity, support_cache_sha256=self.supports.identity)
        root.mkdir(parents=True, exist_ok=True)
        put(root / 'identity.json', scientific)
        cohort = {a['example_id']: a for a in self.assignments[:32]}
        modes = [(m, m.training) for m in self.model.modules()]
        guard = self.model.register_forward_pre_hook(lambda *unused: self.budget.check())
        try:
            self.model.eval()
            rows = {}
            for key in self.inputs:
                self.budget.check()
                if hasattr(self.inputs, '_verify_source'):
                    self.inputs._verify_source(key)
                path = root / ('clean-' + digest(key) + '.json')
                if path.exists() and path.with_name(path.name + '.sha256').exists():
                    row = read_json(path, verify=True)
                    if row['example_id'] != key or row['checkpoint_sha256'] != checksum:
                        raise ValueError('clean dev checkpoint/input mismatch')
                else:
                    item = self.inputs[key]
                    with torch.no_grad(), ExitStack() as stack:
                        health = gate_health(stack, self.native, item, self.supports[key]['payload']['support']) if key in cohort else None
                        logits, ids = self.native('normlens', item)
                        # record_clean consumes the actual raw vocabulary logits from this
                        # existing dev call. Never reconstruct these from prediction files.
                        single = dict(ident, assignments=[cohort[key]]) if key in cohort else None
                        clean = manipulation.record_clean(single, [(key, logits, ids)])['rows'][0] if single else None
                        from .potency import _answer
                        answer = clean['answer'] if clean else _answer(logits, ids)
                    row = dict(example_id=key, checkpoint_sha256=checksum, answer=answer,
                               answer_token_ids=list(ids), clean_record=clean, gate_health=health)
                    put(path, row)
                rows[key] = row
            clean = dict(identity=ident, rows=[rows[a['example_id']]['clean_record'] for a in self.assignments[:32]])
            put(root / 'clean-cohort.json', clean)
            raw_path = root / 'c2-raw.json'
            if raw_path.exists() and raw_path.with_name(raw_path.name + '.sha256').exists():
                raw = read_json(raw_path, verify=True)
            else:
                self.budget.check()
                raw = manipulation.run(self.model, self.inputs, frozen_assignments=self.assignments,
                    clean_predictions=clean, checkpoint_sha256=checksum, seed=self.seed,
                    operator_sha256=self.operator, arm=self.arm, cache_paths=self.paths, pins=self.pins,
                    image_sha256=self.inputs.image_sha256, alignment_ids=self.alignment, dose_protocol=self.dose_protocol)
                put(raw_path, raw)
            joined = manipulation.checkpoint_summary(raw, expected_identity=ident,
                votes=self.inputs.targets, operator_locked=True)
            put(root / 'c2-summary.json', joined)
            nll = -sum(sum(q*p for q,p in zip(self.inputs.targets[key], row['answer']['conditional_log_probabilities']))
                       for key,row in rows.items()) / len(rows)
            put(root / 'gate-health.json', dict(identity=ident, rows={key: rows[key]['gate_health'] for key in cohort}))
            return dict(cond_nll=nll, c2_summary_sha256=digest(joined), c2_directory=str(root))
        except Exception as exc:
            from .pipeline import BudgetExpired
            if isinstance(exc, BudgetExpired):
                return dict(status='paused', reason=str(exc))
            raise
        finally:
            guard.remove()
            for module, mode in modes:
                module.training = mode


class BoundedTrainingRun(TrainingRun):
    """Persist completed updates if the per-forward guard stops the next update."""
    def run(self, max_updates=None):
        from .pipeline import BudgetExpired
        try:
            return super().run(max_updates=max_updates)
        except BudgetExpired:
            if self.next_update and (self.latest_ckpt is None or
                                     self.latest_ckpt['next_update'] != self.next_update):
                self._save(self.next_update, self.trainer.epoch_of(self.next_update-1), 'pause')
            return self._paused('time cap during update; unfinished update discarded')


def _train_model(model, trainer, dev, run_dir, *, budget, resume=False, max_updates=None,
                update_reserve_s, dev_reserve_s):
    """Production callable seam; trainer binds real manifests/schedule before entry."""
    if trainer.identity['protocol'].get('checkpoint_reduction') != REDUCTION:
        raise ValueError('per_family_mean32 must be frozen in run identity before training')
    root = Path(run_dir)
    frozen = frozen_state_sha256(model)
    usable = budget.deadline - __import__('time').monotonic()
    run = BoundedTrainingRun(root, trainer.identity, trainer, dev, budget=RunBudget(max(usable, 1e-9)),
        update_reserve_s=update_reserve_s, dev_reserve_s=dev_reserve_s)
    run.open(resume)
    put(root / 'schedule.json', trainer.schedule)
    put(root / 'frozen-state.json', frozen)
    initial = root / 'initial-dev.json'
    if not initial.exists() or not initial.with_name(initial.name + '.sha256').exists():
        # No saved initial evidence may be fabricated using a resumed trained model.
        if run.next_update:
            raise ValueError('missing initial checkpoint evidence in resumed trained run')
        result = dev(-1, dict(next_update=0, adapter_sha256=trainer.identity['adapter_init']))
        if result.get('status') == 'paused':
            return dict(status='paused', reason='initial dev/C2 pending', next_update=0, run_dir=str(root))
        put(initial, result)
    else:
        read_json(initial, verify=True)
    set_train_mode(model)
    result = run.run(max_updates=max_updates)
    if frozen_state_sha256(model) != frozen:
        raise ValueError('frozen model weights changed')
    expected_positions = [0] + [u+1 for u in range(trainer.total_updates) if trainer.is_epoch_end(u)]
    checkpoints, expected = [], []
    for position in expected_positions:
        path = dev.root / ('u%05d' % position) / 'c2-summary.json'
        if path.exists() and path.with_name(path.name + '.sha256').exists():
            summary = read_json(path, verify=True)
            checkpoints.append(dict(update=position, summary=summary))
            expected.append(dict(update=position, identity=summary['identity'], summary_sha256=digest(summary)))
    # Full prescribed positions are mandatory even if an on-disk partial list is valid.
    report = manipulation.observational_report(run.log, checkpoints, expected_checkpoints=expected,
                                               checkpoint_reduction=REDUCTION)
    if [c['update'] for c in checkpoints] != expected_positions:
        report = dict(status='UNAVAILABLE', observationally_inert=None, reasons=['missing prescribed initial/epoch/final C2 checkpoints'])
    put(root / ('c2-report-u%05d-%s.json' % (run.next_update, digest(report)[:16])), report)
    result.update(run_dir=str(root), c2=report)
    if result['status'] == 'complete':
        result['selection'] = read_json(root / 'selection.json', verify=True) if (root / 'selection.json').exists() else run.select()
    return result


def train_model(model, trainer, dev, run_dir, *, budget, **kwargs):
    """Guard every native training/dev/C2 forward, including callable execution."""
    guard = model.register_forward_pre_hook(lambda *unused: budget.check())
    try:
        return _train_model(model, trainer, dev, run_dir, budget=budget, **kwargs)
    finally:
        guard.remove()


def execute_loaded(model, serializer, launch, train_inputs, *, pins, alignment_ids, budget, resume=False, c2_only=False):
    """Construct the production manifest streams after launch+SealB gates passed."""
    from .data import ManifestInputs
    from .pipeline import validate_index
    from .trainer import HybridTrainer
    from ..schedules import build_training_schedule
    from ..runtime.lora import apply_lora
    from .gates import validate_seal_b
    from ..runtime.pins import load_pins
    seal, artifacts = validate_seal_b(launch, load_pins()[1])
    from .eta050 import launch_protocol
    dose_protocol = launch_protocol(launch)
    data = launch['data']
    dev_inputs = ManifestInputs(data['normlens_manifest'], data['normlens_manifest_sha256'], serializer,
        task='normlens', split='dev', pins=pins, image_dir=data['normlens_image_dir'])
    dev_inputs._identity['launch_provenance'] = train_inputs.identity['launch_provenance']
    nb_inputs = ManifestInputs(data['normbank_manifest'], data['normbank_manifest_sha256'], serializer,
        task='normbank', split='train', pins=pins)
    from .training_cache_source import bind_cache_source
    bind_cache_source(launch, train_inputs, 'train')
    bind_cache_source(launch, dev_inputs, 'dev')
    paths, _ = validate_index(launch['train_cache_index'], train_inputs, pins=pins, alignment_ids=alignment_ids)
    dev_paths, eligible = validate_index(launch['dev_cache_index'], dev_inputs, pins=pins, alignment_ids=alignment_ids)
    from .potency import assignments
    frozen = assignments(eligible)
    potency = artifacts['potency']
    if potency['assignments'] != frozen or potency['pins'] != pins:
        raise ValueError('actual potency cohort/pins differ from training')
    schedule = build_training_schedule(list(train_inputs), list(nb_inputs), launch['seed'], 3, launch['arm'])
    model, _ = apply_lora(model, launch['seed'])
    protocol = dict(seal_b_sha256=launch['seal_b']['sha256'], checkpoint_reduction=REDUCTION,
                    operator_plan_sha256=launch['operator_plan']['sha256'], chronology=seal['chronology'])
    trainer = HybridTrainer(model, schedule, train_inputs, train_inputs.targets, nb_inputs, nb_inputs.targets,
        cache_paths=paths, pins=pins, image_sha256=train_inputs.image_sha256,
        alignment_ids=alignment_ids, protocol_identity=protocol,
        eta=.5 if dose_protocol else .25, dose_protocol=dose_protocol)
    root = Path(launch['paths']['output_root']) / ('hybrid_' + launch['arm'] + '_' + digest(trainer.identity)[:16])
    dev = CheckpointDev(model, dev_inputs, dev_paths, frozen, root / 'dev', pins=pins,
        alignment_ids=alignment_ids, seed=launch['seed'], operator_sha256=launch['operator_plan']['sha256'],
        arm=launch['arm'], budget=budget, dose_protocol=dose_protocol)
    if c2_only and not resume:
        raise ValueError('C2 requires explicit resume of the exact training checkpoint')
    # c2_only resumes and finalizes pending native dev/C2 at the exact checkpoint,
    # then max_updates=0 prevents any optimizer update.
    return train_model(model, trainer, dev, root, budget=budget, resume=resume,
        max_updates=0 if c2_only else None, update_reserve_s=launch['job']['update_reserve_seconds'],
        dev_reserve_s=launch['job']['dev_reserve_seconds'])
