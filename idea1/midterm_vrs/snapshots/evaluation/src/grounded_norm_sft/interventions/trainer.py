"""CPU-callable hybrid SFT using the baseline optimizer and checkpoint contracts."""
from copy import deepcopy
from collections import Counter
from ..runtime.train import B1Trainer, TrainingRun, _finite
from ..runtime.errors import RuntimeContractError
from ..objective import vote_ce
from ..schedules import build_training_schedule
from .cache import digest
from .data import ManifestInputs, SupportInputs
from .forward import NativeForward, validated_supports
from ..constants import EXPECTED_LORA_PARAMS_TOTAL


class HybridTrainer(B1Trainer):
    def __init__(self, model, schedule, normlens_inputs, normlens_targets, normbank_inputs, normbank_targets,
                 *, cache_paths, pins, image_sha256, alignment_ids, protocol_identity, eta=.25,
                 expected_lora_total=EXPECTED_LORA_PARAMS_TOTAL, dose_protocol=None):
        arm = schedule['arm']
        if arm not in ('H-T', 'H-R', 'F-T', 'D-T'):
            raise RuntimeContractError('only H-T/H-R/F-T/D-T are implemented; H-Rmatch is deferred')
        from .eta050 import require_dose
        require_dose(eta, dose_protocol)
        self.dose_protocol = dose_protocol
        if dose_protocol is not None and arm not in ('H-T','H-R'):
            raise ValueError('eta050 amendment authorizes H-T/H-R only')
        if dose_protocol is not None and (schedule['epochs'] != 3 or schedule['run_seed'] not in (17,29,43)):
            raise ValueError('eta050 training preserves three epochs and frozen optimization seeds')
        expected = build_training_schedule(list(normlens_inputs), list(normbank_inputs), schedule['run_seed'],
                                           schedule['epochs'], arm)
        if expected != schedule:
            raise RuntimeContractError('schedule differs from the frozen actual ordinal/site schedule')
        if not protocol_identity or not pins:
            raise ValueError('immutable protocol and model/processor/template pins required')
        from ..runtime.lora import lora_inventory, expected_target_names
        lora_inventory(model, expected_target_names(), expected_lora_total)
        forward = NativeForward(model, alignment_ids=alignment_ids)
        # Own snapshots prevent changes to masks/coordinates while checkpoint graphs are live.
        nl, nb = [x if isinstance(x, ManifestInputs) else deepcopy(x)
                  for x in (normlens_inputs, normbank_inputs)]
        for stream, task, targets in ((nl, 'normlens', normlens_targets), (nb, 'normbank', normbank_targets)):
            if isinstance(stream, ManifestInputs):
                if stream.split != 'train' or stream.task != task:
                    raise ValueError('hybrid training requires the correct task and train manifest split')
                if {k: list(v) for k, v in targets.items()} != {k: list(v) for k, v in stream.targets.items()}:
                    raise ValueError('training targets differ from pinned manifest')
                if any(pins.get(k) != v for k, v in stream.identity['pins'].items()):
                    raise ValueError('serializer pins differ from trainer pins')
        if isinstance(nl, ManifestInputs) and dict(image_sha256) != dict(nl.image_sha256):
            raise ValueError('support image identities differ from pinned raw images')
        support_type = SupportInputs if isinstance(nl, ManifestInputs) else validated_supports
        self.records = support_type(nl, cache_paths, pins=pins, image_sha256=image_sha256,
                                    alignment_ids=alignment_ids)
        forward.bind(nl)
        forward.bind(nb)
        baseline = build_training_schedule(list(nl), list(nb), schedule['run_seed'], schedule['epochs'], 'B1')
        super().__init__([(n, p) for n, p in model.named_parameters() if p.requires_grad], forward, baseline,
                         nl, deepcopy(dict(normlens_targets)), nb, deepcopy(dict(normbank_targets)))
        self.schedule = deepcopy(schedule)
        self.updates = [[deepcopy(r) for r in schedule['rows'] if r['update'] == u]
                        for u in range(self.total_updates)]
        self.arm, self.eta, self.model = arm, eta, model
        if dose_protocol is not None:
            protocol_identity = dict(protocol_identity, dose_protocol=dose_protocol.as_dict())
        self.identity = dict(protocol=deepcopy(protocol_identity), pins=dict(pins), arm=arm, eta=eta,
                             schedule_sha256=digest(self.schedule), support_cache_sha256=(self.records.identity if isinstance(self.records, SupportInputs) else digest(
                                 {k: v['sha256'] for k, v in sorted(self.records.items())})))
        from ..runtime.model import frozen_state_sha256
        from ..runtime.lora import adapter_sha256
        from ..runtime.pins import load_pins
        self.identity.update(frozen_model=frozen_state_sha256(model),
                             adapter_init=adapter_sha256(dict(self.named_params)), runtime_pins=load_pins()[0])
        self._targets_hash = digest(dict(normlens=self.nl_q, normbank=self.nb_q))
        self.identity['targets_sha256'] = self._targets_hash
        self.identity['serialized_inputs_sha256'] = digest(dict(
            eager=sorted(forward._bound.values()), streams=[x.identity for x in forward._streams]))
        self._record_hash = None if isinstance(self.records, SupportInputs) else digest(self.records)
        self._schedule_hash = digest(self.schedule)
        self._exposure = []

    def window_loss_terms(self, rows):
        if digest(dict(normlens=self.nl_q, normbank=self.nb_q)) != self._targets_hash:
            raise ValueError('bound training targets changed')
        if (self._record_hash is not None and digest(self.records) != self._record_hash) or digest(self.schedule) != self._schedule_hash:
            raise ValueError('bound support cache or schedule changed')
        if any(dict(row) != self.schedule['rows'][row['ordinal']] for row in rows):
            raise ValueError('bound update rows changed')
        ce_n, ce_t = [], []
        self._exposure = []
        w = len(rows)
        for row in rows:
            key = row['normlens_id']
            serialized = self.nl_in[key]
            support = self.records[key]['payload']['support']
            active = support['supported'] and self.eta != 0
            logits, ids = self.forward('normlens', serialized)
            _finite(logits, 'clean NormLens logits')
            clean = vote_ce(logits, ids, self.nl_q[key])
            _finite(clean, 'clean NormLens CE')
            (clean * (.5 if active else 1.) / w).backward()
            clean_value = float(clean.detach())
            pert_value, kl = clean_value, 0.
            if active:
                # The scope includes backward, therefore non-reentrant replay sees this exact site/mask.
                with self.forward.perturbed(serialized, support, row, arm=self.arm, eta=self.eta, protocol=self.dose_protocol):
                    changed, changed_ids = self.forward('normlens', serialized)
                    _finite(changed, 'perturbed NormLens logits')
                    pert = vote_ce(changed, changed_ids, self.nl_q[key])
                    _finite(pert, 'perturbed NormLens CE')
                    (.5 * pert / w).backward()
                    pert_value = float(pert.detach())
                    logp = logits.detach().float()[ids].log_softmax(-1)
                    p = logp.exp()
                    logq = changed.detach().float()[ids].log_softmax(-1)
                    kl_tensor = (p * (logp - logq)).sum()
                    _finite(kl_tensor, 'conditional KL diagnostic')
                    kl = float(kl_tensor)
            ce_n.append(.5 * (clean_value + pert_value))
            logits_t, ids_t = self.forward('normbank', self.nb_in[row['normbank_id']])
            _finite(logits_t, 'NormBank logits')
            loss_t = vote_ce(logits_t, ids_t, self.nb_q[row['normbank_id']])
            _finite(loss_t, 'NormBank CE')
            (.25 * loss_t / w).backward()
            ce_t.append(float(loss_t.detach()))
            self._exposure.append(dict(ordinal=row['ordinal'], example_id=key, layer=row['site_layer'],
                                       family=row['site_family'], supported=bool(support['supported']),
                                       duplicate_skipped=not active, conditional_kl=kl,
                                       ce_difference=pert_value-clean_value, abs_ce_difference=abs(pert_value-clean_value)))
        return ce_n, ce_t

    def run_update(self, u):
        from ..runtime.checkpoint import capture_rng, restore_rng
        from .pipeline import BudgetExpired
        rng_t, rng_s = capture_rng()
        try:
            result = super().run_update(u)
        except BudgetExpired:
            # A forward budget stop occurs before optimizer.step; restore RNG
            # and discard unfinished gradients before checkpointing prior updates.
            restore_rng(rng_t, rng_s)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        result['exposure'] = deepcopy(self._exposure)
        result['nominal_sites'] = dict(Counter(str(r['layer']) for r in self._exposure))
        result['supported_sites'] = dict(Counter(str(r['layer']) for r in self._exposure if r['supported']))
        result['duplicate_skips'] = sum(r['duplicate_skipped'] for r in self._exposure)
        return result

    def training_run(self, run_dir, dev_fn, **kwargs):
        """Clean native dev/export callbacks execute after every scoped backward has exited."""
        return TrainingRun(run_dir, self.identity, self, dev_fn, **kwargs)
