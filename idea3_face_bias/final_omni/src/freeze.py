"""Write a PROPOSED freeze for Yash to review (not an approval). Group A: steering, plus base and constitution (no settings).
Test runs additionally need locks/test_approved_A.done, created by hand only after Yash's OK.
Usage: python src/freeze.py A"""
import datetime as dt, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE
import evaluate

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
pct = lambda v: f'{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]'


def main():
    S = json.loads((HERE / 'results/sweep/nemotron_summary.json').read_text()); T, P = S['table'], S['proposal']
    dirs = evaluate.CACHE / 'directions_nemotron.npz'
    rows = sorted((HERE / 'results/sweep').glob('nemotron_*.jsonl'))
    h = {p.name: sha(p) for p in rows}
    J = {'status': 'PROPOSED, not approved', 'steering': {'direction': P['direction'], 'layers': P['layers'], 'alpha': P['alpha']},
         'chosen': P['chosen'], 'fallback_used': P['fallback_used'], 'acc_min': P['acc_min'], 'r_beh_min': P['r_beh_min'],
         'directions_sha256': sha(dirs), 'sweep_rows_sha256': h, 'constitution_sha256': sha(HERE / 'config/constitution.txt'),
         'written': dt.datetime.now().isoformat()[:19]}
    (HERE / 'results/FREEZE_A.json').write_text(json.dumps(J, indent=1))
    L = ['# FREEZE A (PROPOSED): base, constitution, naive-text steering', '',
         'Status: **waiting for Yash\'s review**. No test-split run has been made with these settings. The DPO freeze (group B) comes later; test results of group A stay unread until then.', '',
         f"Rule (START_PROMPT Section 6, fixed before the sweep): lowest validation S_app (text, L2) subject to clean accuracy >= {100 * P['acc_min']:.1f}% and R_beh >= {100 * P['r_beh_min']:.1f}%; ties to smaller alpha, then fewer layers.", '',
         '| Method | Layers | Alpha | Direction | Fallback used |', '|---|---|---|---|---|',
         f"| {P['chosen']} | {P['layers']} | {P['alpha']} | {P['direction']} | {P['fallback_used']} |", '',
         'Validation (196 cue items; %, 95% CI clustered by source video, text channel, L2):', '',
         '| Method | Clean accuracy | S_app | S_0 | R_beh |', '|---|---|---|---|---|']
    for m in ['base'] + sorted(x for x in T if x != 'base'):
        t = T[m]; L.append(f"| {m}{' (chosen)' if m == P['chosen'] else ''} | {pct(t['acc'])} | {pct(t['s_app'])} | {pct(t['s0'])} | {pct(t['r_beh'])} |")
    L += ['', f"Directions SHA-256: `{J['directions_sha256']}`", f"Constitution SHA-256: `{J['constitution_sha256']}`", f"Sweep rows: {len(rows)} files, hashes in FREEZE_A.json"]
    (HERE / 'results/FREEZE_A.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
