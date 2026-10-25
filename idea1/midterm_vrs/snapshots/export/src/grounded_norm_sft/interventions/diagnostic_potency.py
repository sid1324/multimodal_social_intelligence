"""Prospective eta=.5 diagnostic only. Never supplies Seal B or training evidence."""
from contextlib import contextmanager, ExitStack
from statistics import median
import torch
from .native_adapters import _delta, _attention
from .endpoints import delta_endpoint, full_attention_endpoint, _masks
from .scope import one_site
from .potency import assignments, _answer, _r, FAMILIES
from .forward import NativeForward, validated_supports
from .cache import digest
from ..runtime import pins as runtime_pins

AMENDMENT = 'prospective-potency-eta050-active-parity-v1'
DOSE = .5
COHORT_SHA256 = '4ae22d8524cac1304881d41afc2bb3333d748bf7910a30a53d0f366072ba02f5'
ORIGINAL_SHA256 = '89fa0ded30b37a7263b52532116e369444aa6346b93f117372d90cb1376a7b6e'


def frozen(model):
    if torch.is_grad_enabled() or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('diagnostic requires no_grad, eval and frozen parameters')


def mix_half(clean, excluded, rows):
    if clean.shape != excluded.shape or rows.shape != clean.shape[:2]:
        raise ValueError('diagnostic endpoint shape mismatch')
    if not torch.isfinite(clean).all() or not torch.isfinite(excluded).all():
        raise FloatingPointError('nonfinite diagnostic endpoints')
    return torch.where(rows[..., None], (clean.float()+DOSE*(excluded.float()-clean.float())).to(clean.dtype), clean)


@contextmanager
def diagnostic_site(module, *, source, rows, parity=False, diagnostic=None, counter=None, ordinal=0):
    """Always enter custom native preprocessing, including empty-source parity."""
    frozen(module)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    pin = runtime_pins.load_pins()[0]
    runtime_pins.verify_transformers(pin)
    runtime_pins.verify_imported_modeling_module(pin)
    if type(module) is native.Qwen3_5Attention:
        family = 'FA'
        if module.config._attn_implementation != 'eager':
            raise ValueError('diagnostic requires native eager attention')
    elif type(module) is native.Qwen3_5GatedDeltaNet:
        family = 'D'
    else:
        raise TypeError('pinned native mixer required')
    source, rows = source.clone(), rows.clone()
    if not rows.any():
        raise ValueError('nonempty intervention rows required')
    if parity != (not bool(source.any())):
        raise ValueError('parity requires empty source; dose requires nonempty source')
    def bridge(original, *args, **kwargs):
        frozen(module)
        hidden = args[0] if args else kwargs['hidden_states']
        _masks(source, rows, *hidden.shape[:2])
        if source.device != hidden.device or rows.device != hidden.device:
            raise ValueError('native device masks required')
        if counter is not None:
            counter.append(dict(site=module.layer_idx, family=family, parity=parity))
        values = {}
        if family == 'D':
            q,k,v,g,beta,z = _delta(module, native, *args, **kwargs)
            clean = delta_endpoint(q,k,v,g,beta,z, source=source, rows=rows,
                norm=module.norm, out_proj=module.out_proj, kernel=native.torch_chunk_gated_delta_rule,
                diagnostic=lambda **data: values.update(data))
        else:
            y,weights,v,gate = _attention(module, native, *args, **kwargs)
            clean = full_attention_endpoint(y,weights,v,gate, source=source, rows=rows,
                num_key_value_groups=module.num_key_value_groups, repeat_kv=native.repeat_kv,
                out_proj=module.o_proj, diagnostic=lambda **data: values.update(data))
        if parity:
            output = clean
        else:
            output = mix_half(values['clean_output'], values['excluded_output'], rows)
            if diagnostic is not None:
                diagnostic(dict(site=module.layer_idx,family=family,ordinal=ordinal,
                    displacement=_r(values['clean_output'][rows],output[rows]),
                    clean_output_norm=float(values['clean_output'][rows].float().norm()),
                    excluded_output_norm=float(values['excluded_output'][rows].float().norm()),
                    mask_sha256=digest(dict(source=source.cpu().tolist(),rows=rows.cpu().tolist()))))
        return output if family == 'D' else (output, weights)
    with one_site(module, bridge, arm='H-T'):
        yield module


def difference(a,b):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('nonfinite or changed parity tensor shape')
    return float((a.float()-b.float()).abs().max())


def parity_pass(logprob_difference, relative_output, argmax_same):
    return 0 <= logprob_difference <= 1e-6 and 0 <= relative_output <= 1e-5 and argmax_same


def run(model, serialized, *, cache_paths, pins, image_sha256, alignment_ids, check_budget=lambda: None):
    frozen(model)
    if any('lora' in n.lower() or 'adapter' in n.lower() for n,_ in model.named_modules()):
        raise ValueError('diagnostic requires adapter-free base')
    supports = validated_supports(serialized, cache_paths, pins=pins,
        image_sha256=image_sha256, alignment_ids=alignment_ids)
    eligible = [k for k,r in supports.items() if r['payload']['support']['supported']]
    schedule = assignments(eligible)
    if len(schedule) != 128:
        raise ValueError('diagnostic requires exactly fixed 128 eligible assignments')
    if digest(schedule) != COHORT_SHA256:
        raise ValueError('diagnostic cohort differs from original frozen assignments')
    native = NativeForward(model, alignment_ids=alignment_ids)
    native.bind(serialized)
    evidence = dict(amendment=AMENDMENT, dose=DOSE, evidence_kind='prospective_frozen_diagnostic',
        assignments=schedule, cohort_sha256=digest(schedule), original_potency_sha256=ORIGINAL_SHA256,
        parity_criteria=dict(max_abs_full_vocab_logprob=1e-6,max_relative_site_output=1e-5,argmax_mismatches=0),
        pins=pins, rows=[], parity=[], repeats=[],
        intervention_training_allowed=False, seal_b_authorization=False)
    def measured(item):
        check_budget()
        outputs = {}
        def capture(site):
            def hook(module,args,output):
                value = output[0] if isinstance(output,tuple) else output
                outputs[site] = value.detach().clone()
            return hook
        with ExitStack() as stack:
            for site,module in native.sites.items():
                stack.callback(module.register_forward_hook(capture(site)).remove)
            logits,ids = native(item.task,item)
        return logits.detach().clone(), ids, outputs
    for a in schedule[:4]:
        key = a['example_id']; item = serialized[key]
        reference,ids,outputs = measured(item)
        floors = dict(logits=0., logprobs=0., sites={s:0. for s in range(32)}, kl=0.)
        clean = _answer(reference,ids)
        for repeat in range(10):
            logits,rids,current = measured(item)
            if list(rids) != list(ids):
                raise ValueError('repeat answer IDs changed')
            floors['logits'] = max(floors['logits'],difference(reference,logits))
            floors['logprobs'] = max(floors['logprobs'],difference(reference.float().log_softmax(-1),logits.float().log_softmax(-1)))
            floors['kl'] = max(floors['kl'],_answer(logits,ids,clean)['conditional_kl'])
            for s in range(32): floors['sites'][s] = max(floors['sites'][s],difference(outputs[s],current[s]))
        evidence['repeats'].append(dict(example_id=key,repeats=10,floors=floors))
        for site,module in native.sites.items():
            source = torch.zeros((1,len(item.input_ids)),dtype=torch.bool,device=reference.device)
            rows = source.clone(); rows[:,supports[key]['payload']['support']['rows']] = True
            calls = []
            with diagnostic_site(module,source=source,rows=rows,parity=True,counter=calls):
                logits,pids,current = measured(item)
            diffs = dict(logits=difference(reference,logits),
                logprobs=difference(reference.float().log_softmax(-1),logits.float().log_softmax(-1)),
                site=difference(outputs[site],current[site]))
            diffs['relative_site'] = _r(outputs[site],current[site])
            argmax_same = bool(reference.argmax()==logits.argmax())
            passed = len(calls)==1 and list(pids)==list(ids) and parity_pass(diffs['logprobs'],diffs['relative_site'],argmax_same)
            evidence['parity'].append(dict(example_id=key,site=site,custom_calls=len(calls),differences=diffs, argmax_same=argmax_same,bitwise_logits=bool(torch.equal(reference,logits)),passed=passed))
            if not passed:
                evidence['report'] = dict(status='PARITY_FAIL',intervention_training_allowed=False)
                return evidence
    for ordinal,a in enumerate(schedule):
        check_budget()
        key=a['example_id']; item=serialized[key]; support=supports[key]['payload']['support']
        logits,ids = native(item.task,item); clean=_answer(logits,ids)
        grid = {'clean':clean}
        for family in FAMILIES:
            for condition in ('target','random'):
                check_budget()
                source=torch.zeros((1,len(item.input_ids)),dtype=torch.bool,device=logits.device)
                rows=source.clone(); rows[:,support['rows']]=True
                source[:,[support['patches'][i] for i in support['targeted' if condition=='target' else 'random']]]=True
                diagnostics=[]; calls=[]
                with diagnostic_site(native.sites[a[family+'_layer']],source=source,rows=rows,
                        diagnostic=diagnostics.append,counter=calls,ordinal=ordinal):
                    changed,cids=native(item.task,item)
                if len(calls)!=1 or len(diagnostics)!=1 or list(cids)!=list(ids):
                    raise ValueError('one active bridge and endpoint record required')
                probe=_answer(changed,ids,clean); probe.update(eta=DOSE,endpoint=diagnostics[0])
                delta=torch.tensor(probe['answer_logit_changes'])
                raw_clean=torch.tensor(clean['answer_logits'],dtype=torch.float64)
                raw_changed=changed.detach().to(dtype=torch.float64)[list(ids)]
                clp=raw_clean.log_softmax(-1); plp=raw_changed.cpu().log_softmax(-1)
                probe['conditional_kl_float64']=float((clp.exp()*(clp-plp)).sum().clamp_min(0))
                probe['centered_answer_delta_rms']=float((delta-delta.mean()).square().mean().sqrt())
                probe['pairwise_answer_logit_changes']=[float(delta[i]-delta[j]) for i,j in ((0,1),(0,2),(1,2))]
                grid[family+'_'+condition]=probe
        evidence['rows'].append(dict(example_id=key,assignment=a,grid=grid,source_cache_sha256=supports[key]['sha256']))
    families={}
    for family in FAMILIES:
        r=[row['grid'][family+'_target']['endpoint']['displacement'] for row in evidence['rows']]
        kl=[row['grid'][family+'_target']['conditional_kl'] for row in evidence['rows']]
        families[family]=dict(median_r=median(r),median_kl=median(kl),kl_threshold=1e-4,
            passed=median(r)>1e-5 and median(kl)>=1e-4)
    from .potency import _finite
    if not _finite(evidence):
        raise FloatingPointError('nonfinite diagnostic summary')
    evidence['report']=dict(status='PASS' if all(f['passed'] for f in families.values()) else 'FAIL',
        families=families,intervention_training_allowed=False,seal_b_authorization=False)
    return evidence


def execute_loaded(model, launch, inputs, *, pins, alignment_ids, budget):
    """Existing hybrid_potency accounting surrounds this diagnostic-only call."""
    from pathlib import Path
    from .pipeline import validate_index
    from ..hashing import sha256_file
    from ..io_utils import write_once_json
    if launch.get('phase') != 'hybrid_potency' or launch.get('diagnostic_amendment') != AMENDMENT:
        raise ValueError('explicit prospective diagnostic amendment required')
    paths, eligible = validate_index(launch['cache_index'],inputs,pins=pins,alignment_ids=alignment_ids)
    schedule=assignments(eligible)
    clean={a['example_id']:inputs[a['example_id']] for a in schedule}
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    from .pipeline import BudgetExpired
    amendment=launch.get('diagnostic_provenance',{})
    from ..hashing import sha256_file
    if amendment.get('original_potency_sha256') != ORIGINAL_SHA256:
        raise ValueError('original potency artifact binding required')
    for name in ('amendment','user_request'):
        if sha256_file(amendment[name]['path']) != amendment[name]['sha256']:
            raise ValueError('diagnostic provenance bytes mismatch')
    budget.check()
    try:
        with torch.no_grad():
            evidence=run(model,clean,cache_paths={k:paths[k] for k in clean},pins=pins,
                image_sha256={k:inputs.image_sha256[k] for k in clean},alignment_ids=alignment_ids,
                check_budget=budget.check)
    except BudgetExpired:
        write_once_json(Path(launch['paths']['output_root'])/('diagnostic-'+AMENDMENT+'-incomplete.json'),
            dict(amendment=AMENDMENT,status='INCOMPLETE',reason='bounded walltime exhausted',
                 intervention_training_allowed=False,seal_b_authorization=False))
        raise
    evidence['diagnostic_provenance']=amendment
    evidence.update(cache_index_sha256=sha256_file(launch['cache_index']),
        eligible_population_sha256=digest(eligible),launch_sha256=digest(launch))
    path=Path(launch['paths']['output_root'])/('diagnostic-'+AMENDMENT+'.json')
    write_once_json(path,evidence)
    return dict(path=str(path),report=evidence['report'],intervention_training_allowed=False)
