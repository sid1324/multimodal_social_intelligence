"""Pre-DPO check (Nov 24): (1) gradients with the fast MoE dispatch vs the original dispatch, on the real DPO loss;
(2) save/restore of the training state gives the same next steps as an uninterrupted run; (3) seconds per example.
Writes results/gate/dpo_check.json. Train items only. The fast path is used for training only if grad_ok; resume_ok
must be true or training stops (Yash is told).
Usage: python src/dpo_check.py"""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, load_annotations, load_splits, load_item_table, gate_items, permuted_options
from channels import build, scene
import dpo_train as T


def main():
    from omni_model import load
    ann = load_annotations()
    items = gate_items(load_splits(), load_item_table(), 300)[:4]
    ex = []
    for k, cid in enumerate(items):
        ch = ('text', 'speech', 'image', 'text')[k]
        v = ann[cid]; perm = permuted_options(v)
        ex.append((build(v, scene(cid), 'L2_app_low_tr0', ch), perm[1][0], perm[0][0]))
    m = load('nemotron'); pm, ps = T.setup(m)
    yp = torch.tensor([m.letter_ids[e[1]] for e in ex], device=m._w32.device)
    ym = torch.tensor([m.letter_ids[e[2]] for e in ex], device=m._w32.device)
    pm.eval()
    with torch.no_grad(), pm.disable_adapter():
        ref = []
        for e, a, b in zip(ex, yp, ym):
            lp = T.logp_full(m, [e[0]]); ref.append([float(lp[0, a]), float(lp[0, b])])
    ref = torch.tensor(ref, device=m._w32.device)
    T.train_mode(m, pm)

    def grads(seed=0):
        for p in ps:
            p.grad = None
        torch.manual_seed(seed)
        t = time.time()
        for i, e in enumerate(ex):
            loss, _ = T.dpo_loss(T.logp_full(m, [e[0]]), ref[i:i + 1], yp[i:i + 1], ym[i:i + 1])
            (loss.sum() / len(ex)).backward()
        torch.cuda.synchronize()
        return torch.cat([p.grad.detach().flatten().cpu() for p in ps if p.grad is not None]), (time.time() - t) / len(ex)

    def cmp(a, b):
        return {'cos': float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 'rel_norm_diff': float((a.norm() - b.norm()).abs() / b.norm()),
                'max_abs': float((a - b).abs().max()), 'ref_max_abs': float(b.abs().max())}
    for p in ps:   # make grads non-trivial: at init B = 0 so only B gets gradient; nudge all A and B slightly, seeded
        torch.manual_seed(len(p)); p.data.add_(torch.randn_like(p) * 1e-3)
    fast_was = m.fast_moe
    g_fast, t_fast = grads()
    m.disable_fast(); g_orig, t_orig = grads(); g_orig2, _ = grads()
    S = {'fast_vs_orig': cmp(g_fast, g_orig), 'orig_repeat': cmp(g_orig2, g_orig), 'sec_per_example_fast': t_fast,
         'sec_per_example_orig': t_orig, 'fast_enabled_at_load': fast_was, 'gpu': torch.cuda.get_device_name(0),
         'peak_mem_GB': [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]}
    c = S['fast_vs_orig']
    S['grad_ok'] = bool(c['cos'] >= 0.999 and c['rel_norm_diff'] <= 0.02)
    print('GRAD', json.dumps(S), flush=True)

    # resume: 2 steps, snapshot, 2 more; restore snapshot, redo the last 2
    m.enable_fast() if S['grad_ok'] else None
    opt = torch.optim.AdamW(ps, lr=5e-5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    def one(step):
        opt.zero_grad()
        grads(step); opt.step(); sched.step()
    for s in range(2):
        one(s)
    snap = {'params': [p.detach().cpu().clone() for p in ps], 'opt': opt.state_dict(), 'sched': sched.state_dict()}
    snap = torch.load(__import__('io').BytesIO((lambda b: (torch.save(snap, b), b)[1])(__import__('io').BytesIO()).getvalue()), weights_only=True)
    for s in (2, 3):
        one(s)
    A = torch.cat([p.detach().flatten().cpu() for p in ps])
    for p, q in zip(ps, snap['params']):
        p.data.copy_(q.to(p.device))
    opt.load_state_dict(snap['opt']); sched.load_state_dict(snap['sched'])
    for s in (2, 3):
        one(s)
    B = torch.cat([p.detach().flatten().cpu() for p in ps])
    S['resume_max_abs_diff'] = float((A - B).abs().max()); S['resume_param_scale'] = float(A.abs().max())
    S['resume_ok'] = S['resume_max_abs_diff'] < 1e-5
    (HERE / 'results/gate/dpo_check.json').write_text(json.dumps(S, indent=1)); print('DPO_CHECK', json.dumps(S), flush=True)


if __name__ == '__main__':
    main()
