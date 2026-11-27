"""DPO on Gemma 4 12B (port of src/dpo_pairs.py + src/dpo_train.py, which were written for Nemotron). Same settings: config/dpo.yaml.
Question: does a TRAINED fairness fix transfer from typed to spoken remarks? DPO-text (typed pairs only) vs DPO-multi (typed + spoken).

Pairs come from the base model's own answers on training scenes, levels L1 and L2 (as for Nemotron):
  appearance remark changed the answer -> chosen = answer with no remark, rejected = answer with the remark ("ignore looks")
  behavior remark changed the answer   -> chosen = answer with the remark, rejected = answer with no remark ("behavior counts")
Behavior pairs are subsampled to the number of appearance pairs if they outnumber them by more than 1.5x. DPO-multi renders the
same pairs typed and spoken. Single-token DPO on the answer letter, LoRA on the attention projections, frozen reference = the
same model with the adapter off (cached).

Stages (resumable; exit 75 = time limit, run again; nothing touches the test split):
  rows          base answers on the training scenes not in the 300 gate scenes (typed); the gate scenes come from results/gemma
  pairs         build the pairs (CPU)
  train text    DPO-text: typed pairs only                (adapters: results/gemma_dpo/adapters/text/epoch{1,2,3})
  train multi   DPO-multi: the same pairs typed and spoken (adapters: results/gemma_dpo/adapters/multi/epoch{1,2,3})
  eval          every epoch adapter on validation, L1: typed and spoken remarks, plus accuracy on the no-remark scenes
  report        CPU: per adapter and format: appearance flips, floor, behavior flips, accuracy; epoch chosen per variant by the
                midterm rule; DPO-text vs DPO-multi on spoken remarks
Usage: python src/gemma_dpo.py STAGE [text|multi]"""
import json, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, CONFIG, load_annotations, load_splits, load_item_table, gate_items, permuted_options
from channels import build, scene
from evaluate import read_rows, level_conds
from analyze import index, item_stats, summarize, choose, pct, boot
import gemma_speech as gs

R = Path(os.environ.get('GD_OUT', str(HERE / ('results/gemma_dpo' if gs.MODEL == 'gemma' else f'results/{gs.MODEL}_dpo'))))
R.mkdir(parents=True, exist_ok=True)
GEMMA_RESULTS = Path(os.environ.get('GS_RESULTS', str(HERE / ('results/gemma' if gs.MODEL == 'gemma' else f'results/{gs.MODEL}_speech'))))   # this model's speech run (gate rows, base validation rows), if any
CFG = yaml.safe_load((CONFIG / 'dpo.yaml').read_text())
CONDS = [c for lv in CFG['levels'] for c in level_conds(lv, ('app_tr', 'beh_tr'))]
EVAL_LV = os.environ.get('GD_EVAL_LEVEL', 'L1')
N_TRAIN = int(os.environ.get('GD_N_TRAIN', 10 ** 6))   # all; smaller only for the CPU pipeline test
EPOCHS = int(os.environ.get('GD_EPOCHS', CFG['epochs']))
MAX_PAIRS = int(os.environ.get('GD_MAX_PAIRS', 10 ** 9))   # test-only caps; unset in real runs
N_EVAL = int(os.environ.get('GD_N_EVAL', 10 ** 6))


# ---- pairs ---------------------------------------------------------------------------------------------------------
def rows():
    sp = load_splits()
    gate = set(gate_items(sp, load_item_table(), 300))
    have_gate = (GEMMA_RESULTS / 'g2_rows.jsonl').exists()   # without this model's speech run, score the 300 gate scenes here too
    items = [k for k in sp['pool']['train'] if not have_gate or k not in gate][:N_TRAIN]
    cells = [('none', 'text')] + [(c, 'text') for c in CONDS]
    if not gs.grid(R / 'train_rows.jsonl', 'base', items, cells, 'train'):
        sys.exit(75)
    print('rows finished', flush=True)


def pairs():
    sp = load_splits(); train = set(sp['pool']['train'])
    g2p = GEMMA_RESULTS / 'g2_rows.jsonl'
    rr = ([r for r in read_rows(g2p) if r['channel'] == 'text'] if g2p.exists() else []) + read_rows(R / 'train_rows.jsonl')
    pred = {(r['id'], r['condition']): r for r in rr if r['method'] == 'base'}
    items = sorted({i for i, c in pred if c == 'none' and all((i, x) in pred for x in CONDS)})
    assert items and set(items) <= train, 'pairs must come from complete training scenes only'
    out = {'app': [], 'beh': []}
    for cid in items:
        y0 = pred[(cid, 'none')]['pred']
        for c in CONDS:
            y = pred[(cid, c)]['pred']
            if y == y0:
                continue
            fam = 'app' if '_app_' in c else 'beh'
            out[fam].append({'id': cid, 'cond': c, 'family': fam, 'side': c.split('_')[2], 'chosen': y0 if fam == 'app' else y,
                             'rejected': y if fam == 'app' else y0, 'gold': pred[(cid, c)]['gold']})
    raw = {f: len(v) for f, v in out.items()}
    sub = raw['beh'] > CFG['max_behavior_to_appearance'] * raw['app']
    if sub:
        random.Random(CFG['seed']).shuffle(out['beh']); out['beh'] = out['beh'][:raw['app']]
    P = sorted(out['app'] + out['beh'], key=lambda p: (p['id'], p['cond']))
    if len(P) > MAX_PAIRS:
        P = sorted(random.Random(CFG['seed']).sample(P, MAX_PAIRS), key=lambda p: (p['id'], p['cond']))
    cnt = lambda f: dict(sorted(Counter(f(p) for p in P).items()))
    S = {'n_items': len(items), 'raw': raw, 'behavior_subsampled': sub, 'n_pairs': len(P), 'by_family_side': cnt(lambda p: p['family'] + '_' + p['side']),
         'by_level': cnt(lambda p: p['cond'][:2]), 'app_chosen_is_gold': sum(p['chosen'] == p['gold'] for p in P if p['family'] == 'app'),
         'app_rejected_is_gold': sum(p['rejected'] == p['gold'] for p in P if p['family'] == 'app')}
    (R / 'pairs.json').write_text(json.dumps({'summary': S, 'pairs': P}, indent=0))
    print('PAIRS', json.dumps(S), flush=True)


# ---- training ------------------------------------------------------------------------------------------------------
def lora_cfg(m):
    from peft import LoraConfig
    c = CFG['lora']
    return LoraConfig(r=c['r'], lora_alpha=c['alpha'], lora_dropout=c['dropout'], target_modules=m.LORA_REGEX, bias='none')


def logp_full(m, prompts):
    import torch
    hid, _ = m.last_hidden(prompts)
    return torch.log_softmax(m.logits32(hid), -1)


def letters_for(m, v, pair):
    L = {i: l for l, i, _ in permuted_options(v)}
    return m.letter_ids[L[pair['chosen']]], m.letter_ids[L[pair['rejected']]]


def dpo_loss(lp, ref, yp, ym):
    import torch
    margin = (lp.gather(1, yp[:, None])[:, 0] - ref[:, 0]) - (lp.gather(1, ym[:, None])[:, 0] - ref[:, 1])
    return -torch.nn.functional.logsigmoid(CFG['beta'] * margin), margin


def train(variant):
    import torch
    assert variant in ('text', 'multi')
    ann = load_annotations()
    P = json.loads((R / 'pairs.json').read_text())['pairs']
    chans = ('text',) if variant == 'text' else ('text', 'speech')
    ex = [dict(p, channel=ch) for p in P for ch in chans]
    d = R / 'adapters' / variant; d.mkdir(parents=True, exist_ok=True)
    logf = R / f'train_log_{variant}.jsonl'
    m = gs.model()
    torch.manual_seed(CFG['seed'])
    pm = m.add_lora(lora_cfg(m))
    ps = [p for p in pm.parameters() if p.requires_grad]
    for p in ps:
        p.data = p.data.float()
    if hasattr(m.model, 'gradient_checkpointing_enable'):
        m.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(m.model, 'enable_input_require_grads'):
            m.model.enable_input_require_grads()
    dev = m._w32.device
    acc_steps = CFG['effective_batch'] // CFG['micro_batch']
    per_epoch = math.ceil(len(ex) / CFG['effective_batch'])
    total = per_epoch * EPOCHS
    opt = torch.optim.AdamW(ps, lr=CFG['lr'], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step, ck = 0, d / 'last.pt'
    if ck.exists():
        st = torch.load(ck, map_location='cpu', weights_only=True)
        for p, q in zip(ps, st['params']):
            p.data.copy_(q.to(p.device))
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); step = st['step']
    print(f'{variant}: {len(P)} pairs, {len(ex)} examples, {per_epoch} steps/epoch, {total} total, resume at {step}', flush=True)
    key = lambda e: f"{e['id']}|{e['cond']}|{e['channel']}"
    prompt = lambda e: build(ann[e['id']], scene(e['id']), e['cond'], e['channel'])
    rf = d / 'ref.json'
    ref = json.loads(rf.read_text()) if rf.exists() else {}
    todo = [e for e in ex if key(e) not in ref]
    pm.eval()
    with torch.no_grad(), pm.disable_adapter():
        for i, e in enumerate(todo):
            lp = logp_full(m, [prompt(e)])
            yp, ym = letters_for(m, ann[e['id']], e)
            ref[key(e)] = [float(lp[0, yp]), float(lp[0, ym])]
            if i % 200 == 0:
                rf.write_text(json.dumps(ref))
            if gs.timeup():
                rf.write_text(json.dumps(ref)); print('time limit during the reference pass', flush=True); sys.exit(75)
    rf.write_text(json.dumps(ref)); print('reference done', flush=True)

    def save_ck():
        torch.save({'params': [p.detach().cpu() for p in ps], 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, str(ck) + '.tmp')
        os.replace(str(ck) + '.tmp', ck)
    pm.train()
    t0 = time.time()
    while step < total:
        ep = step // per_epoch
        order = list(range(len(ex))); random.Random(CFG['seed'] + ep).shuffle(order)
        torch.manual_seed(CFG['seed'] + step)
        idx = order[(step % per_epoch) * CFG['effective_batch']:][:CFG['effective_batch']]
        opt.zero_grad(); L = M = A = 0.0
        for k in range(0, len(idx), CFG['micro_batch']):
            b = [ex[i] for i in idx[k:k + CFG['micro_batch']]]
            yy = [letters_for(m, ann[e['id']], e) for e in b]
            yp = torch.tensor([y[0] for y in yy], device=dev); ym = torch.tensor([y[1] for y in yy], device=dev)
            r = torch.tensor([ref[key(e)] for e in b], device=dev)
            loss, mar = dpo_loss(logp_full(m, [prompt(e) for e in b]), r, yp, ym)
            (loss.sum() / len(idx)).backward()
            lpd = mar + (r[:, 0] - r[:, 1])
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
        if step % 10 == 0 or step % per_epoch == 0 or gs.timeup():
            save_ck()
        if gs.timeup() and step < total:
            print('time limit at step', step, flush=True); sys.exit(75)
    print(f'TRAIN {variant} DONE', flush=True)


# ---- evaluation ----------------------------------------------------------------------------------------------------
def adapters():
    out = []
    for v in ('text', 'multi'):
        for e in range(1, EPOCHS + 1):
            p = R / 'adapters' / v / f'epoch{e}'
            if (p / 'adapter_model.safetensors').exists():
                out.append((f'dpo_{v}_e{e}', p))
    return out


def eval():
    pool, clean = gs.val_sets()
    pool, clean = pool[:N_EVAL], clean[:N_EVAL]
    cells = [('none', 'text')] + [(c, ch) for ch in ('text', 'speech') for c in level_conds(EVAL_LV)]
    out = R / f'val_{EVAL_LV}.jsonl'
    have_base = any(r['method'] == 'base' for r in read_rows(GEMMA_RESULTS / f'val_{EVAL_LV}.jsonl'))
    m = gs.model()
    if not have_base:   # the speech-transfer run normally provides the base rows; compute them if it does not
        if not (gs.grid(out, 'base', pool, cells, 'val') and gs.grid(out, 'base', clean, [('none', 'text')], 'val')):
            sys.exit(75)
    ad = adapters()
    assert ad, 'no adapters: run train first'
    if gs.os.environ.get('GS_FAKE') == '1':
        runs = [(name, None) for name, _ in ad]
    else:
        from peft import PeftModel
        pm = None
        runs = []
        for name, path in ad:
            if pm is None:
                pm = PeftModel.from_pretrained(m.model, str(path), adapter_name=name)
            else:
                pm.load_adapter(str(path), adapter_name=name)
            runs.append((name, pm))
    for name, pm in runs:
        if pm is not None:
            pm.set_adapter(name); pm.eval()
        if not (gs.grid(out, name, pool, cells, 'val') and gs.grid(out, name, clean, [('none', 'text')], 'val')):
            sys.exit(75)
        print(f'eval {name} finished', flush=True)
    print('EVAL DONE', flush=True)


def report():
    lv = EVAL_LV
    rows = [r for r in read_rows(GEMMA_RESULTS / f'val_{lv}.jsonl') if r['method'] == 'base'] + read_rows(R / f'val_{lv}.jsonl')
    ix = index(rows)
    names = ['base'] + [n for n, _ in adapters() if n in ix]
    T = {m: {ch: summarize({c: item_stats(d, lv, ch) for c, d in ix[m].items()}) for ch in ('text', 'speech')} for m in names}
    for m in names:   # the mean of typed and spoken, used to choose the DPO-multi epoch
        per = {}
        for c, d in ix[m].items():
            ss = [item_stats(d, lv, ch) for ch in ('text', 'speech')]
            per[c] = {k: (None if any(s[k] is None for s in ss) else float(np.mean([s[k] for s in ss]))) for k in ss[0]}
        T[m]['mean'] = summarize(per)
    sel = {}
    for v, ch in (('text', 'text'), ('multi', 'mean')):
        cands = [m for m in names if m.startswith(f'dpo_{v}_')]
        if cands:
            best, flag = choose({m: T[m][ch] for m in T}, T['base'][ch], cands, lambda m: (int(m.rsplit('e', 1)[1]),))
            sel[v] = {'chosen': best, 'selected_on': ch, 'fallback_used': flag}
    md = [f'# DPO on Gemma 4 12B, validation, {lv}', '', 'Epoch chosen per variant by the midterm rule (DPO-text on typed remarks, DPO-multi on the mean of typed and spoken).', '',
          '| Format | Method | Acc, no remark | Appearance flips | Floor | Behavior flips | Change in appearance flips vs base |', '|---|---|---|---|---|---|---|']
    S = {'selection': sel, 'table': T}
    for ch in ('text', 'speech'):
        st = {m: {c: item_stats(d, lv, ch) for c, d in ix[m].items()} for m in names}
        common = set.intersection(*[{c for c, x in st[m].items() if x['s_app'] is not None and x['r_beh'] is not None and x['s0'] is not None} for m in names])
        for m in names:
            t = summarize({c: st[m][c] for c in common})
            dd = None
            if m != 'base':
                byv = defaultdict(list)
                for c in common:
                    byv[c.split('_')[0]].append(st[m][c]['s_app'] - st['base'][c]['s_app'])
                dd = boot(byv)
                S[f'{ch} {m} d_s_app'] = dd
            chosen = any(s['chosen'] == m for s in sel.values())
            md.append(f"| {ch} | {m}{' (chosen)' if chosen else ''} | {pct(T[m]['text']['acc'])} | {pct(t['s_app'])} | {pct(t['s0'])} | {pct(t['r_beh'])} | {pct(dd)} |")
    md += ['', 'Selection: ' + json.dumps(sel)]
    (R / 'report.json').write_text(json.dumps(S, indent=1, default=bool)); (R / 'report.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    st = sys.argv[1]
    globals()[st](*sys.argv[2:])
