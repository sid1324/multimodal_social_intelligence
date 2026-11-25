"""Single-token DPO with LoRA on the LM attention projections (START_PROMPT Section 7).
L = -log sigmoid(beta * [(lp(y+) - lp_ref(y+)) - (lp(y-) - lp_ref(y-))]), lp over the FULL vocabulary at the answer
position; ref = same model with the adapter disabled (computed once per pair and cached; the reference is constant).
Resumable: the adapter, optimizer, scheduler and step are saved every 20 optimizer steps and at each epoch end.
Usage: python src/dpo_train.py --model nemotron --variant text|multi"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, CONFIG, P777, load_annotations, permuted_options
from channels import build, scene
from evaluate import DEADLINE

CFG = yaml.safe_load((CONFIG / 'dpo.yaml').read_text())
ADAPT = Path(P777) / 'adapters'


def lora_cfg(m):
    from peft import LoraConfig
    c = CFG['lora']
    return LoraConfig(r=c['r'], lora_alpha=c['alpha'], lora_dropout=c['dropout'], target_modules=m.LORA_REGEX, bias='none')


def adapter_dirs(model):
    """[(method name, path)] for every saved epoch checkpoint."""
    out = []
    for v in ('text', 'multi'):
        for e in range(1, CFG['epochs'] + 1):
            p = ADAPT / f'{model}_{v}/epoch{e}'
            if (p / 'adapter_model.safetensors').exists():
                out.append((f'dpo_{v}_e{e}', str(p)))
    return out


def train_mode(m, peft_model):
    """eval() everywhere (so the Mamba kernels take the same path as at inference), but train() for the LoRA dropout and
    the block-level checkpointing flag."""
    peft_model.eval()
    for n, mod in peft_model.named_modules():
        if 'lora_dropout' in n:
            mod.train()
    for b in m.layers:
        b.training = True     # flag only: enables gradient checkpointing in the block wrapper, not in the mixers


def setup(m):
    torch.manual_seed(CFG['seed'])
    pm = m.add_lora(lora_cfg(m))
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    base = m.lm
    for mod in base.modules():     # the remote code's blocks are GradientCheckpointingLayers, only the class flag is missing
        if hasattr(type(mod), 'supports_gradient_checkpointing'):
            type(mod).supports_gradient_checkpointing = True
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    train_mode(m, pm)
    ps = [p for p in pm.parameters() if p.requires_grad]
    return pm, ps


def logp_full(m, prompts):
    hid, _ = m.last_hidden(prompts)
    return torch.log_softmax(m.logits32(hid), -1)


def letters_for(m, v, pair):
    L = {i: l for l, i, _ in permuted_options(v)}
    return m.letter_ids[L[pair['chosen']]], m.letter_ids[L[pair['rejected']]]


def dpo_loss(lp, ref, yp, ym):
    margin = (lp.gather(1, yp[:, None])[:, 0] - ref[:, 0]) - (lp.gather(1, ym[:, None])[:, 0] - ref[:, 1])
    return -torch.nn.functional.logsigmoid(CFG['beta'] * margin), margin


def smoke(m, items, ann, steps=10):
    """G1(d): 10 steps on 4 fixed toy pairs (chosen = second letter, rejected = first); the loss must fall."""
    ex = []
    for cid in items[:4]:
        v = ann[cid]
        perm = permuted_options(v)
        ex.append((build(v, scene(cid), 'none', 'text'), m.letter_ids[perm[1][0]], m.letter_ids[perm[0][0]]))
    pm, ps = setup(m)
    opt = torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.0)
    yp = torch.tensor([e[1] for e in ex], device=m._w32.device); ym = torch.tensor([e[2] for e in ex], device=m._w32.device)
    with torch.no_grad(), pm.disable_adapter():
        lp = torch.cat([logp_full(m, [ex[i][0] for i in (2 * j, 2 * j + 1)]) for j in range(2)])
        ref = torch.stack([lp.gather(1, yp[:, None])[:, 0], lp.gather(1, ym[:, None])[:, 0]], 1)
    losses = []
    t = time.time()
    for s in range(steps):
        opt.zero_grad(); tot = 0.0
        for j in range(2):
            sl = slice(2 * j, 2 * j + 2)
            loss, _ = dpo_loss(logp_full(m, [e[0] for e in ex[sl]]), ref[sl], yp[sl], ym[sl])
            (loss.mean() / 2).backward(); tot += float(loss.mean()) / 2
        opt.step(); losses.append(tot)
        print('smoke', s, tot, flush=True)
    gn = sum(float(p.grad.norm()) ** 2 for p in ps if p.grad is not None) ** 0.5
    return {'losses': losses, 'n_trainable': sum(p.numel() for p in ps), 'n_tensors': len(ps), 'grad_norm': gn,
            'sec_per_pair': (time.time() - t) / (steps * 4), 'peak_mem_GB': [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())],
            'moved': losses[-1] < losses[0] - 1e-3 and all(math.isfinite(x) for x in losses)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='nemotron'); ap.add_argument('--variant', choices=['text', 'multi'], required=True)
    a = ap.parse_args()
    from omni_model import load
    ann = load_annotations()
    pairs = json.loads((HERE / f'results/dpo/{a.model}_pairs.json').read_text())['pairs']
    chans = ('text',) if a.variant == 'text' else ('text', 'speech', 'image')
    ex = [dict(p, channel=ch) for p in pairs for ch in chans]
    d = ADAPT / f'{a.model}_{a.variant}'; d.mkdir(parents=True, exist_ok=True)
    logf = HERE / f'results/dpo/{a.model}_{a.variant}_train_log.jsonl'
    chk = json.loads((HERE / 'results/gate/dpo_check.json').read_text())
    assert chk['resume_ok'], 'dpo_check: resume failed'
    m = load(a.model); pm, ps = setup(m)
    if not chk['grad_ok']:
        m.disable_fast(); print('fast MoE off for training: gradient check failed', flush=True)
    acc_steps = CFG['effective_batch'] // CFG['micro_batch']
    per_epoch = math.ceil(len(ex) / CFG['effective_batch'])
    total = per_epoch * CFG['epochs']
    opt = torch.optim.AdamW(ps, lr=CFG['lr'], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step, ck = 0, d / 'last.pt'
    if ck.exists():
        st = torch.load(ck, map_location='cpu', weights_only=True)
        for p, q in zip(ps, st['params']):
            p.data.copy_(q.to(p.device))
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); step = st['step']
    print(f'{len(pairs)} pairs, {len(ex)} examples, {per_epoch} steps/epoch, {total} total, resume at {step}', flush=True)

    # reference log-probs (adapter disabled), cached and resumable
    rf = d / 'ref.json'
    ref = json.loads(rf.read_text()) if rf.exists() else {}
    key = lambda e: f"{e['id']}|{e['cond']}|{e['channel']}"
    def prompt(e):
        return build(ann[e['id']], scene(e['id']), e['cond'], e['channel'])
    def save_ck():
        torch.save({'params': [p.detach().cpu() for p in ps], 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, str(ck) + '.tmp')
        os.replace(str(ck) + '.tmp', ck)
    todo = [e for e in ex if key(e) not in ref]
    pm.eval()
    with torch.no_grad(), pm.disable_adapter():
        for i in range(0, len(todo), 1):   # one at a time, as in scoring
            b = todo[i:i + 1]
            lp = logp_full(m, [prompt(e) for e in b])
            for j, e in enumerate(b):
                yp, ym = letters_for(m, ann[e['id']], e)
                ref[key(e)] = [float(lp[j, yp]), float(lp[j, ym])]
            if i % 200 == 0:
                rf.write_text(json.dumps(ref))
            if time.time() > DEADLINE:
                rf.write_text(json.dumps(ref)); sys.exit(75)
    rf.write_text(json.dumps(ref)); print('reference done', flush=True)
    train_mode(m, pm)
    dev, t0 = m._w32.device, time.time()
    while step < total:
        ep = step // per_epoch
        order = list(range(len(ex))); random.Random(CFG['seed'] + ep).shuffle(order)
        torch.manual_seed(CFG['seed'] + step)
        idx = order[(step % per_epoch) * CFG['effective_batch']:][:CFG['effective_batch']]
        opt.zero_grad(); L = M = A = 0.0
        # group by channel-length similarity is unnecessary: micro-batches of 2 are right-padded
        for k in range(0, len(idx), CFG['micro_batch']):
            b = [ex[i] for i in idx[k:k + CFG['micro_batch']]]
            yy = [letters_for(m, ann[e['id']], e) for e in b]
            yp = torch.tensor([y[0] for y in yy], device=dev); ym = torch.tensor([y[1] for y in yy], device=dev)
            r = torch.tensor([ref[key(e)] for e in b], device=dev)
            loss, mar = dpo_loss(logp_full(m, [prompt(e) for e in b]), r, yp, ym)
            (loss.sum() / len(idx)).backward()
            lpd = mar + (r[:, 0] - r[:, 1])          # lp(y+) - lp(y-): >0 means the chosen letter is preferred
            L += float(loss.sum()); M += float(mar.sum()); A += float((lpd > 0).sum())
        g = float(torch.nn.utils.clip_grad_norm_(ps, 1.0))
        opt.step(); sched.step(); step += 1
        rec = {'step': step, 'epoch': ep + 1, 'loss': L / len(idx), 'margin': M / len(idx), 'acc': A / len(idx), 'grad_norm': g,
               'lr': sched.get_last_lr()[0], 't': round(time.time() - t0)}
        assert math.isfinite(rec['loss']), rec
        with open(logf, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        if step % 5 == 0:
            print(rec, flush=True)
        if step % per_epoch == 0:
            pm.save_pretrained(d / f'epoch{step // per_epoch}')
        if step % 10 == 0 or step % per_epoch == 0 or time.time() > DEADLINE:
            save_ck()
        if time.time() > DEADLINE and step < total:
            print('deadline', step, flush=True); sys.exit(75)
    print('DPO DONE', flush=True)


if __name__ == '__main__':
    main()
