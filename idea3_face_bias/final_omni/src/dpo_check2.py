"""Resume check, version 3 (Nov 25; version 2 compared against a reference its own restore had modified through an aliased step counter, so it read 2.0; pass rule unchanged). Version 1's limit of 1e-5 ignored that the same forward pass is not bit-repeatable on
this model. Pass rule fixed BEFORE running: (1) right after restoring from the saved FILE (same code path as dpo_train),
every parameter and every optimizer and scheduler value equals the snapshot exactly; (2) the parameters after the next two
steps differ from the uninterrupted run by at most 2x the difference between two uninterrupted repeats of those steps.
Also times the fast and original MoE paths after warm-up. Writes results/gate/dpo_check.json (gradient result copied from v1)."""
import copy, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, P777, load_annotations, load_splits, load_item_table, gate_items, permuted_options
from channels import build, scene
import dpo_train as T


def main():
    from omni_model import load
    v1 = json.loads((HERE / 'results/gate/dpo_check_v1.json').read_text())
    ann = load_annotations()
    items = gate_items(load_splits(), load_item_table(), 300)[:4]
    ex = []
    for k, cid in enumerate(items):
        v = ann[cid]; perm = permuted_options(v)
        ex.append((build(v, scene(cid), 'L2_app_low_tr0', ('text', 'speech', 'image', 'text')[k]), perm[1][0], perm[0][0]))
    m = load('nemotron'); pm, ps = T.setup(m)
    yp = torch.tensor([m.letter_ids[e[1]] for e in ex], device=m._w32.device); ym = torch.tensor([m.letter_ids[e[2]] for e in ex], device=m._w32.device)
    pm.eval()
    with torch.no_grad(), pm.disable_adapter():
        ref = []
        for e, a, b in zip(ex, yp, ym):
            lp = T.logp_full(m, [e[0]]); ref.append([float(lp[0, a]), float(lp[0, b])])
    ref = torch.tensor(ref, device=m._w32.device); T.train_mode(m, pm)
    for p in ps:
        torch.manual_seed(len(p)); p.data.add_(torch.randn_like(p) * 1e-3)

    def grads(seed=0):
        for p in ps:
            p.grad = None
        torch.manual_seed(seed); t = time.time()
        for i, e in enumerate(ex):
            loss, _ = T.dpo_loss(T.logp_full(m, [e[0]]), ref[i:i + 1], yp[i:i + 1], ym[i:i + 1]); (loss.sum() / len(ex)).backward()
        torch.cuda.synchronize(); return (time.time() - t) / len(ex)
    S = dict(v1, version=3, gpu=torch.cuda.get_device_name(0))
    grads(); S['sec_per_example_fast'] = sum(grads() for _ in range(2)) / 2
    m.disable_fast(); grads(); S['sec_per_example_orig'] = grads(); m.enable_fast()
    print('TIMING after warm-up', S['sec_per_example_fast'], S['sec_per_example_orig'], flush=True)

    opt = torch.optim.AdamW(ps, lr=5e-5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    def one(step):
        opt.zero_grad(); grads(step); opt.step(); sched.step()
    def flat():
        return torch.cat([p.detach().flatten().cpu() for p in ps])
    def state():
        return {'params': [p.detach().cpu().clone() for p in ps], 'opt': copy.deepcopy(opt.state_dict()), 'sched': copy.deepcopy(sched.state_dict())}
    def restore(st):
        for p, q in zip(ps, st['params']):
            p.data.copy_(q.to(p.device))
        opt.load_state_dict(copy.deepcopy(st['opt'])); sched.load_state_dict(copy.deepcopy(st['sched']))   # deep copies: v2 shared the step counter with its own reference
    one(0); one(1)
    mem = state()                                  # exact in-memory copy
    f = Path(P777) / 'adapters/_resume_check.pt'; f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'params': [p.detach().cpu() for p in ps], 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': 2}, str(f))
    one(2); one(3); A = flat()
    restore(mem); one(2); one(3); A2 = flat()      # repeat of the same steps: noise
    st = torch.load(f, map_location='cpu', weights_only=True); restore(st)
    exact_p = float(max((p.detach().cpu() - q).abs().max() for p, q in zip(ps, mem['params'])))
    exact_o = 0.0
    for i, s in opt.state_dict()['state'].items():
        for k, x in s.items():
            y = mem['opt']['state'][i][k]
            exact_o = max(exact_o, float((torch.as_tensor(x).cpu() - torch.as_tensor(y).cpu()).abs().max()))
    exact_s = int(sched.state_dict()['last_epoch'] != mem['sched']['last_epoch'])
    one(2); one(3); B = flat()
    noise, diff = float((A - A2).abs().max()), float((A - B).abs().max())
    S.update(restore_param_diff=exact_p, restore_opt_diff=exact_o, restore_sched_mismatch=exact_s, noise_max_abs=noise,
             noise_mean_abs=float((A - A2).abs().mean()), resume_max_abs_diff=diff, resume_mean_abs_diff=float((A - B).abs().mean()))
    S['resume_ok'] = bool(exact_p == 0 and exact_o == 0 and exact_s == 0 and (diff <= 2 * noise or diff <= 1e-6))
    f.unlink()
    (HERE / 'results/gate/dpo_check.json').write_text(json.dumps(S, indent=1)); print('DPO_CHECK3', json.dumps(S), flush=True)


if __name__ == '__main__':
    main()
