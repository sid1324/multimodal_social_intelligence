"""Stage 5 test grid (START_PROMPT). Runs ONLY when Yash approved the relevant freeze: ALLOW_TEST=yash-approved must be
set, the method's frozen settings must be in results/FREEZE_<group>.json, and the matching approval marker in
locks/test_approved_<group>.done must exist (created by hand after Yash's OK). Test once per method.
Groups: A = base, constitution, naive-text steering (staged freeze). B = DPO methods.
Usage: run_test.py METHOD SHARD   (METHOD in base, constitution, naive_text, dpo_text, dpo_multi; SHARD 0..3)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, P777, constitution, load_splits
import evaluate
from evaluate import run_grid, level_conds

NSHARD = 4
LOCKS = Path(P777) / 'locks'


def cells(method, pool_only=False):
    M = [('none', 'text')]
    for ch in ('text', 'speech', 'image'):
        M += [(c, ch) for c in level_conds('L2', ('neutral', 'app_tr', 'app_ho', 'beh_ho'))]
    T = [(f'L2_state_{s}{i}', 'text') for s in ('neg', 'pos') for i in (0, 1)]
    T += [(c, 'text') for c in level_conds('L1', ('app_ho', 'beh_ho'))]
    T += [(c, 'text') for c in level_conds('L3', ('neutral', 'app_ho', 'beh_ho'))]
    return M + T


def main():
    method, shard = sys.argv[1], int(sys.argv[2])
    group = 'A' if method in ('base', 'constitution', 'naive_text') else 'B'
    assert os.environ.get('ALLOW_TEST') == 'yash-approved', 'ALLOW_TEST not set: refusing'
    assert (LOCKS / f'test_approved_{group}.done').exists(), f'group {group} not approved by Yash'
    frz = json.loads((HERE / f'results/FREEZE_{group}.json').read_text())
    sp = load_splits(); pool = sp['pool']['test'][shard::NSHARD]
    from omni_model import load
    m = load('nemotron')
    system = None
    if method == 'constitution':
        system = constitution()
    elif method == 'naive_text':
        s = frz['steering']
        m.set_ablation(np.load(evaluate.CACHE / 'directions_nemotron.npz')[s['direction']], float(s['alpha']), s['layers'])
    elif method.startswith('dpo_'):
        from peft import PeftModel
        pm = PeftModel.from_pretrained(m.lm, frz[method]['adapter'], adapter_name='x'); pm.set_adapter('x'); pm.eval()
    out = HERE / f'results/test/nemotron_{method}_s{shard}.jsonl'
    ok = run_grid(m, out, method, pool, cells(method), 'test', system)
    # accuracy set (clean items, no remark); shard 0 only, excluding items already covered above
    if ok and shard == 0:
        clean = [k for k in sp['clean']['test'] if k not in set(sp['pool']['test'])]
        ok = run_grid(m, HERE / f'results/test/nemotron_{method}_clean.jsonl', method, clean, [('none', 'text')], 'test', system)
    sys.exit(0 if ok else 75)


if __name__ == '__main__':
    main()
