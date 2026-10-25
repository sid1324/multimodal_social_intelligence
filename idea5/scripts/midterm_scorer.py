"""Answer EgoNormia from one context embedding: zero-shot cosine and a trained diagonal-bilinear scorer.

  s(c, o) = alpha * c.o + sum_d u_d c_d o_d + [v.o] + beta * 1[o is the empty option]
  L = -mean_i [log p(y_i | actions) + log p(y_i | justifications)] + lam (|u|^2 + |v|^2)
The "lowrank" variant swaps u and v for a rank-R bilinear term (A c).(B o), penalised by lam (|A|^2 + |B|^2): it
can learn scene-option interactions the diagonal u cannot (u stays near zero and reproduces zero-shot).
alpha is a free temperature on the cosine, u a regularised per-dimension reweighting. The option-prior term v.o
scores options without looking at the scene; the "scene" variant drops it so that only scene-option matching can
answer, which isolates what the context vector contributes.
lam is chosen by 5-fold CV grouped by source video inside the training items only.

Writes output/idea5-midterm/results/<cond>.json with per-item predictions for
  test: train on train+train_ext, predict the 200 verified items
  cv:   5-fold CV grouped by source video over all 800 items
  zero_shot: cosine argmax on the test items
usage: midterm_scorer.py C0 C1 ... | --selftest
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
OUT = ROOT / 'output/idea5-midterm'
EMB = OUT / 'emb'
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
SEEDS = 5
EPOCHS = 300


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def folds(ids, k=5):
    sources = sorted({i.split('_')[0] for i in ids}, key=lambda s: digest('fold:' + s))
    fold_of = {s: n % k for n, s in enumerate(sources)}
    return np.array([fold_of[i.split('_')[0]] for i in ids])


VARIANT = 'full'  # 'full' (with option prior), 'scene' (no option prior) or 'lowrank' (bilinear, no option prior)
RANK = 32


class Scorer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(20.0))
        self.u = torch.nn.Parameter(torch.zeros(dim))
        self.v = torch.nn.Parameter(torch.zeros(dim), requires_grad=VARIANT == 'full')
        self.beta = torch.nn.Parameter(torch.zeros(1))
        if VARIANT == 'lowrank':
            self.u.requires_grad_(False)
            self.A = torch.nn.Parameter(torch.randn(RANK, dim) * 0.01)
            self.B = torch.nn.Parameter(torch.randn(RANK, dim) * 0.01)

    def forward(self, c, o, empty):
        # c: n x d, o: n x 5 x d, empty: n x 5
        s = torch.einsum('nd,nmd->nm', c * (self.alpha + self.u), o) + o @ self.v + self.beta * empty
        if VARIANT == 'lowrank':
            s = s + torch.einsum('nr,nmr->nm', c @ self.A.T, o @ self.B.T)
        return s

    def penalty(self):
        if VARIANT == 'lowrank':
            return (self.A ** 2).sum() + (self.B ** 2).sum()
        return (self.u ** 2).sum() + (self.v ** 2).sum()


def fit(c, oa, oj, ea, ej, y, lam, seed):
    torch.manual_seed(seed)
    model = Scorer(c.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ce(model(c, oa, ea), y) + ce(model(c, oj, ej), y) + lam * model.penalty()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict(model, c, oa, oj, ea, ej):
    return model(c, oa, ea).argmax(1).numpy(), model(c, oj, ej).argmax(1).numpy()


def ensemble(c_tr, tr, c_te, te, lam):
    """Majority vote over seeds (ties go to the lowest seed's choice)."""
    votes_a, votes_j = [], []
    for seed in range(SEEDS):
        m = fit(c_tr, *tr, lam, seed)
        a, j = predict(m, c_te, *te)
        votes_a.append(a)
        votes_j.append(j)
    va, vj = np.stack(votes_a), np.stack(votes_j)
    mode = lambda v: np.array([np.bincount(col, minlength=5).argmax() for col in v.T])  # noqa: E731
    return mode(va), mode(vj)


def choose_lambda(c, opts, y, ids):
    f = folds(ids)
    best = None
    for lam in LAMBDAS:
        correct = 0
        for k in range(5):
            tr, te = f != k, f == k
            m = fit(c[tr], *[t[tr] for t in opts], y[tr], lam, 0)
            a, j = predict(m, c[te], *[t[te] for t in opts])
            correct += int(((a == y[te].numpy()) & (j == y[te].numpy())).sum())
        if best is None or correct > best[1]:
            best = (lam, correct)
    return best[0]


def load(cond, ids, data):
    keep = [i for i in ids if (EMB / cond / f'{i}.npy').exists() and (EMB / 'options_action' / f'{i}.npy').exists()
            and (EMB / 'options_justification' / f'{i}.npy').exists()]
    c = torch.tensor(np.stack([np.load(EMB / cond / f'{i}.npy') for i in keep]).astype(np.float32))
    oa = torch.tensor(np.stack([np.load(EMB / 'options_action' / f'{i}.npy') for i in keep]).astype(np.float32))
    oj = torch.tensor(np.stack([np.load(EMB / 'options_justification' / f'{i}.npy') for i in keep]).astype(np.float32))
    ea = torch.tensor([[float(not t) for t in data[i]['behaviors']] for i in keep])
    ej = torch.tensor([[float(not t) for t in data[i]['justifications']] for i in keep])
    y = torch.tensor([data[i]['correct'] for i in keep])
    return keep, c, (oa, oj, ea, ej), y


def run(cond, split, data):
    test_ids, train_ids = split['test'], split['train'] + split['train_ext']
    k_tr, c_tr, o_tr, y_tr = load(cond, train_ids, data)
    k_te, c_te, o_te, y_te = load(cond, test_ids, data)
    lam = choose_lambda(c_tr, o_tr, y_tr, k_tr)
    a, j = ensemble(c_tr, (*o_tr, y_tr), c_te, o_te, lam)
    # Zero-shot cosine over the non-empty options.
    za = (torch.einsum('nd,nmd->nm', c_te, o_te[0]) - 1e9 * o_te[2]).argmax(1).numpy()
    zj = (torch.einsum('nd,nmd->nm', c_te, o_te[1]) - 1e9 * o_te[3]).argmax(1).numpy()
    # Grouped 5-fold CV over everything, lambda fixed to the one chosen on train.
    k_all, c_all, o_all, y_all = load(cond, test_ids + train_ids, data)
    f = folds(k_all)
    cv_a, cv_j = np.zeros(len(k_all), int), np.zeros(len(k_all), int)
    for k in range(5):
        tr, te = f != k, f == k
        cv_a[te], cv_j[te] = ensemble(c_all[tr], (*[t[tr] for t in o_all], y_all[tr]), c_all[te], [t[te] for t in o_all], lam)
    rows = lambda ids, pa, pj: {i: {'action': int(x), 'justification': int(z), 'gold': int(data[i]['correct'])}  # noqa: E731
                                for i, x, z in zip(ids, pa, pj)}
    result = {'cond': cond, 'lambda': lam, 'n_train': len(k_tr), 'n_test': len(k_te), 'n_cv': len(k_all),
              'test': rows(k_te, a, j), 'zero_shot': rows(k_te, za, zj), 'cv': rows(k_all, cv_a, cv_j)}
    (OUT / 'results').mkdir(exist_ok=True)
    (OUT / 'results' / f'{cond}_{VARIANT}.json').write_text(json.dumps(result))
    joint = lambda r: np.mean([v['action'] == v['gold'] and v['justification'] == v['gold'] for v in r.values()])  # noqa: E731
    print(f"{cond} [{VARIANT}]: lam={lam} test joint={joint(result['test']):.3f} (n={len(k_te)}) zero-shot={joint(result['zero_shot']):.3f} "
          f"cv joint={joint(result['cv']):.3f} (n={len(k_all)})", flush=True)


def selftest():
    """Planted signal: the gold option embedding is shifted toward the context; the scorer must find it."""
    rng = np.random.default_rng(0)
    n, d = 300, 64
    c = rng.normal(size=(n, d)).astype(np.float32)
    o = rng.normal(size=(n, 5, d)).astype(np.float32)
    y = rng.integers(0, 4, n)
    o[np.arange(n), y] += 0.3 * c * (np.arange(d) < 8)  # signal only in 8 dims, invisible to plain cosine
    t = lambda x: torch.tensor(x)  # noqa: E731
    empty = torch.zeros(n, 5)
    empty[:, 4] = 1
    tr, te = slice(0, 200), slice(200, None)
    m = fit(t(c[tr]), t(o[tr]), t(o[tr]), empty[tr], empty[tr], t(y[tr]), 1e-3, 0)
    a, _ = predict(m, t(c[te]), t(o[te]), t(o[te]), empty[te], empty[te])
    cos = np.einsum('nd,nmd->nm', c[te], o[te])[:, :4].argmax(1)
    print('selftest trained acc', (a == y[te]).mean(), 'cosine acc', (cos == y[te]).mean(), 'chance 0.25')


def main():
    if sys.argv[1:] == ['--selftest']:
        return selftest()
    data = json.loads((DATA / 'annotations/final_data.json').read_text())
    split = json.loads((OUT / 'splits.json').read_text())
    global VARIANT
    args = sys.argv[1:]
    variants = ['full', 'scene']
    if args and args[0].startswith('--variant='):
        variants = [args.pop(0).split('=', 1)[1]]
    for cond in args:
        for VARIANT in variants:
            run(cond, split, data)


if __name__ == '__main__':
    main()
