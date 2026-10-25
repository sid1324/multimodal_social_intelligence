"""Bounded, pre-Seal-B native acceptance. Never selects scientific checkpoints.

All model forwards use fixed TRAIN records. The caller disposes this model before
profiling/training. Local operator oracles use activations captured from that same
model, its real head shapes, native kernels and endpoint weights.
"""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from .cache import digest
from ..hashing import sha256_file
from ..io_utils import write_once_json

CHECKS = ('native_inventory_alignment', 'source_algebra', 'prefix_isolation',
          'endpoint_gradients', 'checkpointing_sequential_equivalence',
          'same_environment_resume', 'native_disabled_repeats', 'bf16_native_reference')


def validate_spec(spec):
    if set(spec) != {'fixture_ids', 'normbank_fixture_id', 'max_prompt_tokens', 'checks'}:
        raise ValueError('explicit native_acceptance fixture IDs, token bound and all checks required')
    ids = spec['fixture_ids']
    if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
        raise ValueError('exactly one fixed TRAIN NormLens fixture required')
    if not isinstance(spec['normbank_fixture_id'], str) or not spec['normbank_fixture_id']:
        raise ValueError('fixed TRAIN NormBank fixture required')
    if type(spec['max_prompt_tokens']) is not int or not 1 <= spec['max_prompt_tokens'] <= 512:
        raise ValueError('native acceptance prompt cap must be 1..512; truncation forbidden')
    if spec['checks'] != list(CHECKS):
        raise ValueError('all eight native acceptance checks in declared order required')


def compare(actual, expected, *, atol, rtol, nrms=None):
    """Record errors and fail at the frozen tolerance, including tiny magnitudes."""
    if set(actual) != set(expected) or not actual:
        raise ValueError('comparison tensor inventories differ or are empty')
    maximum, rms_max = 0., 0.
    for key in actual:
        dtype=torch.float64 if actual[key].dtype==torch.float64 or expected[key].dtype==torch.float64 else torch.float32
        a, b = actual[key].detach().to(dtype).cpu(), expected[key].detach().to(dtype).cpu()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError('nonfinite acceptance tensor: '+str(key))
        error = a-b
        maximum = max(maximum, float(error.abs().max()))
        rms = float(error.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8))
        rms_max = max(rms_max, rms)
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        if nrms is not None and rms > nrms:
            raise ValueError('normalized RMS exceeds frozen tolerance: '+str(key))
    return dict(max_abs_error=maximum, max_normalized_rms_error=rms_max,
                atol=atol, rtol=rtol, normalized_rms_limit=nrms, tensors=len(actual))


def _snapshot(trainer, gradients=False):
    return {n: (p.grad if gradients and p.grad is not None else torch.zeros_like(p) if gradients else p)
            .detach().cpu().clone() for n,p in trainer.named_params}


def operator_checks(captured, sites, budget, *, eta=.25, protocol=None):
    """Independent recurrent Delta oracle on native-scale, actual head-shape inputs.

    Captured q/k are already normalized/repeated by native preprocessing. Both
    scans and the nonlinear gated normalization/projection remain differentiable.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from .native_adapters import _delta
    from .endpoints import delta_endpoint, interpolate
    module = sites[0]
    args, kwargs = captured[0]
    with torch.no_grad():
        values = _delta(module, native, *args, **kwargs)
    q,k,v,g,beta,z = [x[:, :8].detach() for x in values]
    source = torch.zeros(v.shape[:2], device=v.device, dtype=torch.bool)
    source[:, 1:3] = True
    rows = torch.zeros_like(source); rows[:, 4:] = True
    algebra, gradients, bf16 = [], [], []
    for scale in (1., 1e-4):
        budget.check()
        xs = [(x*scale if i==2 else x).detach().requires_grad_()
              for i,x in enumerate((q,k,v,g,beta,z))]
        qq,kk,vv,gg,bb,zz = xs
        def scan(value, kernel):
            return kernel(qq,kk,value,g=gg,beta=bb,use_qk_l2norm_in_kernel=True)[0]
        def transform(y):
            return module.out_proj(module.norm(y.reshape(-1,y.shape[-1]),
                zz.reshape(-1,zz.shape[-1])).reshape(y.shape[0],y.shape[1],-1))
        kernel = native.torch_chunk_gated_delta_rule
        y = scan(vv,kernel)
        chosen = scan(vv*source[...,None,None],kernel)
        excluded = scan(vv.masked_fill(source[...,None,None],0),kernel)
        # These are BF16 actual activations on production; FP32 fixtures retain strict FP32 limits.
        low = vv.dtype == torch.bfloat16
        out_tol = dict(atol=.02,rtol=.02,nrms=.02) if low else dict(atol=2e-5,rtol=2e-4)
        grad_tol = dict(atol=.05,rtol=.05,nrms=.05) if low else dict(atol=5e-5,rtol=5e-4)
        algebra.append(compare({'readout':y},{'readout':chosen+excluded},**out_tol))
        actual = delta_endpoint(*xs,source=source,rows=rows,norm=module.norm,
            out_proj=module.out_proj,kernel=kernel,eta=eta,protocol=protocol)
        clean = transform(scan(vv,native.torch_recurrent_gated_delta_rule))
        ex = transform(scan(vv.masked_fill(source[...,None,None],0),native.torch_recurrent_gated_delta_rule))
        reference = torch.where(rows[...,None],(clean.float()+eta*(ex.float()-clean.float())).to(clean.dtype),clean)
        output = compare({'output':actual},{'output':reference},**out_tol)
        ga = torch.autograd.grad(actual.float().square().sum(),xs,retain_graph=True)
        gb = torch.autograd.grad(reference.float().square().sum(),xs,retain_graph=True)
        gradient = compare(dict(enumerate(ga)),dict(enumerate(gb)),**grad_tol)
        wrong = interpolate(clean,ex.detach(),rows,eta,protocol=protocol)
        detached = torch.autograd.grad(wrong.float().square().sum(),xs,allow_unused=True)
        wrong_error = max(float((a.float()-(b.float() if b is not None else 0)).abs().max())
                          for a,b in zip(gb,detached))
        if wrong_error == 0:
            raise ValueError('detached-excluded negative control was not detected')
        gradients.append(dict(scale=scale, parity=gradient, detached_negative_control_max_error=wrong_error))
        bf16.append(dict(scale=scale, dtype=str(vv.dtype), output=output, gradient=gradient,
                         shape=list(vv.shape)))
    from .native_adapters import _attention
    from .endpoints import full_attention_endpoint
    fa=sites[3]
    with torch.no_grad():
        _,coeff,value,gate=_attention(fa,native,*captured[3][0],**captured[3][1])
    for scale in (1.,1e-4):
        weights=coeff[:,:,:8,:8].detach().requires_grad_()
        value=(value[:,:,:8]*scale).detach().requires_grad_()
        gating=gate[:,:8].detach().requires_grad_()
        expanded=native.repeat_kv(value,fa.num_key_value_groups)
        clean=(weights@expanded).transpose(1,2).contiguous()
        actual=full_attention_endpoint(clean,weights,value,gating,source=source,rows=rows,
            num_key_value_groups=fa.num_key_value_groups,repeat_kv=native.repeat_kv,
            out_proj=fa.o_proj,eta=eta,protocol=protocol)
        # Independent headwise contraction, retaining live coefficients and gate.
        excluded=torch.einsum('bhts,bhsd->bthd',weights,
            expanded.masked_fill(source[:,None,:,None],0))
        def project(y):return fa.o_proj(y.reshape(1,8,-1)*gating.sigmoid())
        a,b=project(clean),project(excluded)
        expected=torch.where(rows[...,None],(a.float()+eta*(b.float()-a.float())).to(a.dtype),a)
        tolerance=dict(atol=.02,rtol=.02,nrms=.02) if value.dtype==torch.bfloat16 else dict(atol=2e-5,rtol=2e-4)
        out=compare({'FA':actual},{'FA':expected},**tolerance)
        inputs=(weights,value,gating)
        ga=torch.autograd.grad(actual.float().square().sum(),inputs,retain_graph=True)
        gb=torch.autograd.grad(expected.float().square().sum(),inputs)
        tolerance=dict(atol=.05,rtol=.05,nrms=.05) if value.dtype==torch.bfloat16 else dict(atol=5e-5,rtol=5e-4)
        grad=compare(dict(enumerate(ga)),dict(enumerate(gb)),**tolerance)
        gradients.append(dict(family='FA',scale=scale,parity=grad))
        bf16.append(dict(family='FA',scale=scale,dtype=str(value.dtype),output=out,gradient=grad))
    # Zero/one/random masks, short sequences and right-padding on actual repeated heads.
    for length in (1, 3, 8):
        for mode in ('zero', 'one', 'alternating', 'right_padding'):
            budget.check()
            qq,kk,vv,gg,bb=[x[:,:length] for x in (q,k,v,g,beta)]
            mask=torch.zeros(vv.shape[:2],device=vv.device,dtype=torch.bool)
            if mode=='one':mask[:]=True
            elif mode=='alternating':mask[:,::2]=True
            elif mode=='right_padding':
                mask[:,:max(1,length//2)]=True
                vv=vv.clone(); vv[:,max(1,length//2):]=0
            def component(value):
                return native.torch_chunk_gated_delta_rule(qq,kk,value,g=gg,beta=bb,
                    initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
            a,sa=component(vv);b,sb=component(vv*mask[...,None,None]);c,sc=component(vv*(~mask[...,None,None]))
            tolerance=dict(atol=.02,rtol=.02,nrms=.02) if vv.dtype==torch.bfloat16 else dict(atol=2e-5,rtol=2e-4)
            algebra.append(dict(length=length,mask=mode,comparison=compare(
                {'output':a,'state':sa},{'output':b+c,'state':sb+sc},**tolerance)))
    gradients.append(dict(float64_finite_difference=finite_difference_oracle(q,k,v,g,beta,z,eta)))
    return dict(source_algebra=dict(fixtures=algebra, zero_initial_state=True),
                endpoint_gradients=dict(fixtures=gradients), bf16_native_reference=dict(fixtures=bf16))


def _run_checks(trainer, *, output, identity, budget, fixture_mode=False):
    """Execute on ONE disposable model; snapshots live on CPU and no weights persist."""
    from ..runtime.model import set_train_mode, delta_dispatch_inventory
    from ..runtime.checkpoint import (capture_rng, restore_rng, save_checkpoint,
        load_checkpoint, resume_trainer, optimizer_to_tensors)
    from ..objective import vote_ce
    from .native_adapters import native_site
    model, forward = trainer.model, trainer.forward
    key = next(iter(trainer.nl_in)); item = trainer.nl_in[key]
    support = trainer.records[key]['payload']['support']
    if not support['supported'] or not support['targeted']:
        raise ValueError('fixed train fixture must have nonempty supported cached patches')
    if len(trainer.nl_in)!=1 or len(trainer.nb_in)!=1 or trainer.total_updates<2:
        raise ValueError('bounded acceptance requires one fixed example/task and at least two updates')
    if not fixture_mode:
        from ..runtime.model import check_device
        from ..runtime.lora import production_expectation
        check_device()
        production_expectation(model.config.to_dict())
        if not any(p.dtype==torch.bfloat16 for p in model.parameters()):
            raise ValueError('real native acceptance requires pinned BF16 backbone')
    eos=model.config.text_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    if any(eos.intersection(x.answer_token_ids) for x in [*trainer.nl_in.values(),*trainer.nb_in.values()]):
        raise ValueError('EOS cannot be an acceptance SFT answer target')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    checks = {}
    def record(name, measurements):
        data = dict(check=name, identity=identity, measurements=measurements, fixture_mode=fixture_mode)
        path = output/(name+'.json'); write_once_json(path,data)
        checks[name] = dict(exit_code=0, command='hybrid profile-preseal: native_acceptance.'+name,
            measurements=measurements, evidence=dict(path=str(path),sha256=sha256_file(path)))
    record('native_inventory_alignment',dict(trainer_identity=trainer.identity,
        fixture_ids=list(trainer.nl_in),normbank_ids=list(trainer.nb_in),
        sequence_tokens=len(item.input_ids),alignment_ids=forward.alignment_ids,
        support_sha256=digest(support),dispatch=delta_dispatch_inventory(),
        sites={str(k):type(v).__name__ for k,v in forward.sites.items()},no_eos_target=True))
    model.eval()
    source=torch.zeros((1,len(item.input_ids)),device=next(model.parameters()).device,dtype=torch.bool)
    source[0,[support['patches'][i] for i in support['targeted']]]=True
    rows=torch.zeros_like(source); rows[0,support['rows']]=True
    prefix=~rows[0]
    captured={}
    def capture(site):
        def hook(module,args,kwargs):
            if site not in captured:
                # Native preprocessing is evaluated later with these exact tensors.
                captured[site]=(args,kwargs)
        return hook
    def observe(serialized=item):
        result={}
        def hook(i):
            def save(module,args,value):
                x=value[0] if isinstance(value,tuple) else value
                result[str(i)]=x[0].detach().cpu().clone()
            return save
        with ExitStack() as stack:
            from ..runtime.model import base_of
            for i,module in enumerate(base_of(model).model.language_model.layers):
                h=module.register_forward_hook(hook(i)); stack.callback(h.remove)
            with torch.no_grad():
                logits,_=forward('normlens',serialized)
        result['logits']=logits.detach().cpu().clone()
        return result
    with ExitStack() as stack:
        for site in (0,3):
            h=forward.sites[site].register_forward_pre_hook(capture(site),with_kwargs=True);stack.callback(h.remove)
        untouched=observe()
    # Measured native floor is independent of disabled or perturbed differences.
    floor=0.
    for repeat in range(10):
        budget.check(); result=observe()
        floor=max(floor,max(float((result[k].float()-untouched[k].float()).abs().max()) for k in result))
    disabled=[]
    for site in (0,3):
        for repeat in range(10):
            budget.check()
            with native_site(forward.sites[site],arm='H-T',source=source,rows=rows,enabled=False):
                disabled.append(compare(observe(),untouched,atol=floor,rtol=0))
    bypass=[]
    for site in (0,3):
        for name,mask,eta,protocol in [('eta_zero',source,0,None),
                ('empty_support',torch.zeros_like(source),trainer.eta,trainer.dose_protocol)]:
            with native_site(forward.sites[site],arm='H-T',source=mask,rows=rows,eta=eta,protocol=protocol):
                bypass.append(dict(site=site,bypass=name,comparison=compare(observe(),untouched,atol=floor,rtol=0)))
    record('native_disabled_repeats',dict(bypass=bypass,untouched_repeats=10,disabled_repeats_per_family=10,
        max_native_repeat_absolute_floor=floor,bitwise_native_stable=floor==0,comparisons=disabled))
    isolation=[]
    for site in (0,3):
        budget.check()
        with native_site(forward.sites[site],arm='H-T',source=source,rows=rows,
                         eta=trainer.eta,protocol=trainer.dose_protocol):
            changed=observe()
        isolation.append(dict(site=site,comparison=compare({k:v if int(k)<site else v[prefix.cpu()] for k,v in changed.items() if k!='logits'},
            {k:v if int(k)<site else v[prefix.cpu()] for k,v in untouched.items() if k!='logits'},atol=floor,rtol=0)))
    from dataclasses import replace
    # Synthetic suffix change after the completed action; no labels or new images.
    variant_ids=list(item.input_ids)
    at=item.completed_action_row+1
    protected=set(forward.alignment_ids.values())
    candidates=[i for i in range(at,len(variant_ids)) if variant_ids[i] not in protected]
    choices=[token for token in variant_ids[at:] if token not in protected]
    changed_positions=[]
    for i in candidates:
        other=next((token for token in choices if token!=variant_ids[i]),None)
        if other is not None:
            variant_ids[i]=other;changed_positions.append(i)
            if len(changed_positions)==2:break
    if not changed_positions or variant_ids[:at]!=item.input_ids[:at]:
        raise ValueError('synthetic future-instruction variant construction failed')
    inputs=dict(item.model_inputs); inputs['input_ids']=[variant_ids]
    variant=replace(item,input_ids=variant_ids,model_inputs=inputs)
    future=observe(variant)
    future_parity=compare({k:v[:at] for k,v in future.items() if k!='logits'},
        {k:v[:at] for k,v in untouched.items() if k!='logits'},atol=floor,rtol=0)
    record('prefix_isolation',dict(pre_R_rows=int(prefix.sum()),all_32_decoder_layer_outputs=isolation,
        future_suffix_changed_positions=changed_positions,identical_prefix_through_action=True,
        completed_action_and_earlier_outputs=future_parity))
    local=operator_checks(captured,forward.sites,budget,eta=trainer.eta,protocol=trainer.dose_protocol)
    for name,value in local.items():record(name,value)
    del captured, untouched, disabled
    rng=capture_rng()
    set_train_mode(model)
    grad_repeats=[]
    def gradients(site,checkpoint,sequential,eta):
        budget.check(); trainer.optimizer.zero_grad(set_to_none=True);restore_rng(*rng)
        if checkpoint:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:model.gradient_checkpointing_disable()
        row=dict(trainer.updates[0][0],site_layer=site,site_family='D' if site==0 else 'FA')
        a,ids=forward('normlens',item);a=vote_ce(a,ids,trainer.nl_q[key])*.5
        if sequential:a.backward()
        from contextlib import nullcontext
        context=(nullcontext() if eta is None else forward.perturbed(item,support,row,arm=trainer.arm,
            eta=eta,protocol=trainer.dose_protocol if eta else None))
        with context:
            b,ids=forward('normlens',item);b=vote_ce(b,ids,trainer.nl_q[key])*.5
            if sequential:b.backward()
            else:(a+b).backward()
        return _snapshot(trainer,True)
    low=any(p.dtype==torch.bfloat16 for p in model.parameters())
    tol=dict(atol=.05,rtol=.05,nrms=.05) if low else dict(atol=5e-5,rtol=5e-4)
    for site in (0,3):
        base=gradients(site,False,True,trainer.eta)
        repeated=gradients(site,False,True,trainer.eta)
        variability=compare(repeated,base,**tol)
        checked=compare(gradients(site,True,True,trainer.eta),base,**tol)
        summed=compare(gradients(site,False,False,trainer.eta),base,**tol)
        zero=gradients(site,False,True,0)
        native=gradients(site,False,False,None)
        native_repeat=compare(gradients(site,False,False,None),native,**tol)
        grad_repeats.append(dict(site=site,repeated_perturbed_gradient_variability=variability,repeated_native_gradient_variability=native_repeat,
            checkpointing=checked,sequential_summed=summed,eta_zero=compare(zero,native,**tol)))
    record('checkpointing_sequential_equivalence',dict(families=grad_repeats))
    model.gradient_checkpointing_disable();trainer.optimizer.zero_grad(set_to_none=True);restore_rng(*rng)
    budget.check();trainer.run_update(0)
    with TemporaryDirectory(prefix='acceptance-resume-') as temp:
        saved=save_checkpoint(Path(temp)/'checkpoint',trainer=trainer,next_update=1,epoch=trainer.epoch_of(0),
            kind='disposable_native_acceptance',identity=trainer.identity,log=[],extra=dict(scientific_trained_artifact=False))
        checkpoint=load_checkpoint(saved['dir'],temp)
        budget.check();trainer.run_update(1)
        expected=_snapshot(trainer);expected_opt,_=optimizer_to_tensors(trainer.optimizer,trainer.param_names)
        expected_scheduler=trainer.scheduler.state_dict();expected_rng=capture_rng()
        # New optimizer/scheduler objects ensure restoration is independent of
        # the already advanced live optimizer. Reuse the single frozen backbone.
        from copy import copy
        from ..runtime.train import build_optimizer, build_scheduler
        trainer=copy(trainer)
        trainer.optimizer=build_optimizer(trainer.params)
        trainer.scheduler=build_scheduler(trainer.optimizer,trainer.total_updates)
        trainer._exposure=[]
        cursor=resume_trainer(trainer,checkpoint,trainer.identity)
        budget.check();trainer.run_update(cursor)
        measured=compare(_snapshot(trainer),expected,**tol)
        actual_opt,_=optimizer_to_tensors(trainer.optimizer,trainer.param_names)
        optim=compare(actual_opt,expected_opt,**tol)
        actual_rng=capture_rng()
        if expected_scheduler!=trainer.scheduler.state_dict() or expected_rng[1]!=actual_rng[1]:
            raise ValueError('resume scheduler/RNG metadata differs')
        rng_comparison=compare(actual_rng[0],expected_rng[0],atol=0,rtol=0)
        record('same_environment_resume',dict(adapter=measured,optimizer=optim,rng=rng_comparison,
            checkpoint_manifest_sha256=saved['manifest_sha256'],cursor=cursor,identity=trainer.identity,
            checkpoint_deleted=True,scientific_checkpoint=False,
            reload_scope='fresh trainer shell and optimizer/scheduler, same single frozen backbone'))
    receipt=dict(schema='gn_hybrid_native_acceptance_v1',**identity,checks={k:checks[k] for k in CHECKS},
        evidence_kind='tiny_native_fixture' if fixture_mode else 'real_9b_native_training_acceptance',
        fixture_mode=fixture_mode,scientific_trained_artifact=False)
    path=output/'acceptance.json';write_once_json(path,receipt)
    return dict(path=str(path),sha256=sha256_file(path))


def run_checks(trainer, *, output, identity, budget, fixture_mode=False):
    guard = trainer.model.register_forward_pre_hook(lambda *unused: budget.check())
    try:
        return _run_checks(trainer,output=output,identity=identity,budget=budget,fixture_mode=fixture_mode)
    finally:
        guard.remove()


def finite_difference_oracle(q,k,v,g,beta,z,eta):
    """Float64 algebra oracle with centered differences through both scans/endpoints."""
    values=[x[:,:4,:1,:2].detach().cpu().double() for x in (q,k,v,z)]
    qq,kk,vv,zz=values
    gg=g[:,:4,:1].detach().cpu().double();bb=beta[:,:4,:1].detach().cpu().double()
    vv.requires_grad_()
    def loss(value):
        def scan(value):
            query=torch.nn.functional.normalize(qq,dim=-1)*qq.shape[-1]**-.5
            key=torch.nn.functional.normalize(kk,dim=-1)
            state=torch.zeros((*key.shape[:1],1,2,2),dtype=torch.float64)
            outputs=[]
            for i in range(value.shape[1]):
                state=state*gg[:,i].exp()[...,None,None]
                delta=(value[:,i]-(state*key[:,i,...,None]).sum(-2))*bb[:,i,...,None]
                state=state+key[:,i,...,None]*delta[...,None,:]
                outputs.append((state*query[:,i,...,None]).sum(-2))
            return torch.stack(outputs,dim=1)
        source=torch.tensor([False,True,False,False])[None,:,None,None]
        def endpoint(y):
            return y*torch.rsqrt(y.square().mean(-1,keepdim=True)+3e-6)*torch.nn.functional.silu(zz)
        clean=endpoint(scan(value));excluded=endpoint(scan(value.masked_fill(source,0)))
        rows=torch.tensor([False,False,True,True])[None,:,None,None]
        out=torch.where(rows,clean+eta*(excluded-clean),clean)
        return out.square().sum()
    analytic=torch.autograd.grad(loss(vv),vv)[0]
    numeric=torch.empty(vv.shape,dtype=torch.float64);eps=1e-5
    for i in range(vv.numel()):
        plus=vv.detach().clone().contiguous();minus=vv.detach().clone().contiguous()
        plus.reshape(-1)[i]+=eps;minus.reshape(-1)[i]-=eps
        numeric.reshape(-1)[i]=(loss(plus)-loss(minus))/(2*eps)
    return compare({'value':analytic},{'value':numeric},atol=1e-5,rtol=1e-3)
