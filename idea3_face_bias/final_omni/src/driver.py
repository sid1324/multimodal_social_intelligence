"""Work-queue driver: every job runs this. It takes the next task whose prerequisites (and gates) are met, with lock
files so that jobs in different partitions never do the same work. All tasks are resumable. Gates are enforced here:
 - speech/image G2 and everything later require the text G2 at L2 to pass (else results/STOP.md is written);
 - they also require the perception check >= 90% in every channel.
No test-split task exists (data.load_splits hides it). Usage: python src/driver.py"""
import json, os, re, subprocess, sys, threading, time, traceback
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE
import evaluate
from evaluate import DEADLINE, read_rows
import omni_model

LOCKS = Path(os.environ['P777']) / 'locks'; LOCKS.mkdir(exist_ok=True)
G = HERE / 'results/gate'
STATE = {'model': None, 'name': None, 'dirty': False}
_load = omni_model.load


def load(name):
    if STATE['name'] != name:
        assert STATE['model'] is None, 'one model per process'
        STATE['model'], STATE['name'] = _load(name), name
    m = STATE['model']; m.set_ablation(None)
    return m


omni_model.load = load


def done_file(t):
    return LOCKS / f'{t}.done'


def is_done(t):
    return done_file(t).exists()


def stop(msg):
    (HERE / 'results/STOP.md').write_text(msg + '\n'); print('STOP', msg, flush=True)


# ---- task bodies (GPU) ------------------------------------------------------------------------
def t_moecheck():
    import run_gate; run_gate.moecheck(NS())


def t_perception():
    import run_gate; run_gate.perception(NS(model='nemotron'))


def t_g2(part, shard):
    import run_gate; run_gate.g2(NS(model='nemotron', level='L2', part=part, shard=shard))


def t_sweep(name):
    import numpy as np
    from data import load_splits
    sp = load_splits(); pool = sp['pool']['val']; clean = [k for k in sp['clean']['val'] if k not in set(pool)]
    m = load('nemotron')
    if name != 'base':
        nm, al = name[len('naive_'):].rsplit('_a', 1)
        u = np.load(evaluate.CACHE / 'directions_nemotron.npz')['app_text_L2']
        m.set_ablation(u, float(al), evaluate.layer_sets(m.n_layers)[nm])
    cells = [('none', 'text')] + [(c, 'text') for c in evaluate.level_conds('L2')]
    out = HERE / f'results/sweep/nemotron_{name}.jsonl'
    ok = evaluate.run_grid(m, out, name, pool, cells, 'val') and evaluate.run_grid(m, out, name, clean, [('none', 'text')], 'val')
    sys.exit(0 if ok else 75)


def t_g2L1(shard):
    """Exploratory (Yash, Nov 24): speech and image at L1 on the same 300 gate items."""
    from data import load_splits, load_item_table, gate_items
    items = gate_items(load_splits(), load_item_table(), 300)[shard::2]
    cells = [(c, ch) for ch in ('speech', 'image') for c in evaluate.level_conds('L1')]
    ok = evaluate.run_grid(load('nemotron'), G / f'g2_nemotron_otherL1_s{shard}.jsonl', 'base', items, cells, 'train')
    sys.exit(0 if ok else 75)


def t_blind(variant):
    """Does the scene matter? (Yash, Nov 24) Same prompts, with a gray image of the same size (blank) or the scene of a
    different source video (wrong). 300 gate items, train split; L2 text cues and the no-remark condition."""
    from PIL import Image
    from data import load_splits, load_item_table, gate_items, seeded_int, video_of
    real = evaluate.scene
    items = gate_items(load_splits(), load_item_table(), 300)
    if variant == 'blank':
        evaluate.scene = lambda cid: Image.new('RGB', real(cid).size, (127, 127, 127))
    else:
        order = sorted(items, key=lambda k: seeded_int('blind-wrong-v1', k))
        n = len(order); other = {}
        for i, k in enumerate(order):     # next item in a seeded order that comes from another video
            j = next(j for j in range(1, n) if video_of(order[(i + j) % n]) != video_of(k))
            other[k] = order[(i + j) % n]
        evaluate.scene = lambda cid: real(other[cid])
    cells = [('none', 'text')] + [(c, 'text') for c in evaluate.level_conds('L2', ('neutral', 'app_tr'))]
    try:
        ok = evaluate.run_grid(load('nemotron'), G / f'blind_nemotron_{variant}.jsonl', f'base_{variant}', items, cells, 'train')
    finally:
        evaluate.scene = real
    sys.exit(0 if ok else 75)


def t_preflight():
    """New machine check: rescore 120 stored G2 prompts (train items) and compare answers with the stored Delta rows."""
    import torch
    from data import load_annotations
    from channels import build, scene
    from data import permuted_options
    rows = [r for p in sorted(G.glob('g2_nemotron_*_s?.jsonl')) for r in read_rows(p)]
    store = {(r['id'], r['channel'], r['condition']): r for r in rows}
    ids = sorted({r['id'] for r in rows})[:20]
    cells = [('none', 'text'), ('L2_neutral', 'text'), ('L2_app_low_tr0', 'text'), ('L2_app_high_tr0', 'text'), ('L2_app_low_tr0', 'speech'), ('L2_app_low_tr0', 'image')]
    ann = load_annotations(); m = load('nemotron')
    ok, dps, n, t0 = 0, [], 0, 0
    for k, cid in enumerate(ids):
        v, img = ann[cid], scene(cid)
        if k == 2:
            torch.cuda.synchronize(); t0 = time.time()
        for c, ch in cells:
            res = m.score_many([build(v, img, c, ch)], len(permuted_options(v)))[0][0]
            r = evaluate.row_for(cid, v, c, ch, res, 'train', 'pre', 'nemotron'); s0 = store[(cid, ch, c)]
            ok += r['pred'] == s0['pred']; n += 1
            import math
            dps.append(max(abs(math.exp(r['logp'][o]) - math.exp(s0['logp'][o])) for o in r['logp']))
    sec = (time.time() - t0) / (len(cells) * (len(ids) - 2))
    S = {'n': n, 'agree': ok / n, 'mean_max_dp': sum(dps) / n, 'max_dp': max(dps), 'sec_per_pass': sec, 'fast_moe': m.fast_moe,
         'gpu': torch.cuda.get_device_name(0), 'n_gpu': torch.cuda.device_count(), 'peak_GB': round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    S['pass'] = S['agree'] >= 0.95
    (G / 'preflight.json').write_text(json.dumps(S, indent=1)); print('PREFLIGHT', json.dumps(S), flush=True)
    if not S['pass']:
        stop(f'Preflight: this machine agrees with the stored rows on only {ok}/{n}. Yash and Claude should look before spending time.')
        sys.exit(1)


def t_dpo_rows():
    import dpo_pairs; dpo_pairs.run(NS(model='nemotron'))


def sub(*cmd):
    """DPO tasks change the model, so they run in their own process."""
    r = subprocess.run([sys.executable, *cmd], cwd=HERE)
    sys.exit(r.returncode)


SW = ['base'] + [f'naive_{n}_a{a}' for n in evaluate.layer_sets(52) for a in (0.5, 1.0)]


# ---- CPU transitions ----------------------------------------------------------------------------
def py(*cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=HERE, capture_output=True, text=True)
    print(r.stdout[-6000:], r.stderr[-1500:], flush=True)
    return r.returncode == 0


def transitions():
    if is_done('g2_text_s0') and is_done('g2_text_s1') and not is_done('gate_text'):
        if py('src/analyze.py', 'g2', '--model', 'nemotron'):
            s = json.loads((G / 'g2_nemotron_summary.json').read_text())
            if s['text L2']['pass']:
                done_file('gate_text').touch()
            elif not (HERE / 'results/STOP.md').exists():
                stop('G2: L2 text shows no appearance effect above its floor. Yash must pick L1 or L3 (START_PROMPT G2).')
    if is_done('perception') and not is_done('gate_perc'):
        s = json.loads((G / 'perception_nemotron.json').read_text())
        if min(s.values()) >= 0.9:
            done_file('gate_perc').touch()
        elif not (HERE / 'results/STOP.md').exists():
            stop(f'Perception check below 90%: {s}')
    if all(is_done(f'g2_{p}_s{k}') for p in ('text', 'other') for k in (0, 1)) and not is_done('directions'):
        if py('src/analyze.py', 'g2', '--model', 'nemotron') and py('src/directions.py', '--model', 'nemotron', '--level', 'L2'):
            done_file('directions').touch(); py('src/figures.py', 'f2')
    if is_done('dpo_rows') and not is_done('pairs') and py('src/dpo_pairs.py', 'build', '--model', 'nemotron'):
        done_file('pairs').touch()


def tasks():
    """(name, ready, body, needs_own_process) in priority order."""
    gate = is_done('gate_text') and is_done('gate_perc')
    pre = os.environ.get('I3_PREFLIGHT') == '1'
    out = [('preflight', pre, t_preflight, False)] if pre else []
    out += [('moecheck3', True, t_moecheck, False), ('perception', True, t_perception, False)]
    for k in (0, 1):
        out.append((f'g2_text_s{k}', True, lambda k=k: t_g2('text', k), False))
    for k in (0, 1):
        out.append((f'g2_other_s{k}', gate, lambda k=k: t_g2('other', k), False))
    sweeps = [(f'sweep_{n}', is_done('directions'), lambda n=n: t_sweep(n), False) for n in SW]
    out += [(f'blind_{x}', gate, lambda x=x: t_blind(x), False) for x in ('blank', 'wrong')]
    out += sweeps[:1] + [('dpo_rows', is_done('directions'), t_dpo_rows, False), ('dpo_check3', is_done('dpo_check2') and ('SLURM_JOB_END_TIME' not in os.environ or time.time() < DEADLINE - 480), lambda: sub('src/dpo_check2.py'), True)] + sweeps[1:]   # dpo_rows is base scoring only
    out += [(f'g2L1_s{k}', gate, lambda k=k: t_g2L1(k), False) for k in (0, 1)]
    out += [('dpo_text', is_done('pairs') and is_done('dpo_check3') and dpo_ok(), lambda: sub('src/dpo_train.py', '--model', 'nemotron', '--variant', 'text'), True),
            ('dpo_multi', is_done('pairs') and is_done('dpo_check3') and dpo_ok(), lambda: sub('src/dpo_train.py', '--model', 'nemotron', '--variant', 'multi'), True)]
    for grp, ms in (('A', ('base', 'constitution', 'naive_text')), ('B', ('dpo_text', 'dpo_multi'))):
        ok_test = (LOCKS / f'test_approved_{grp}.done').exists() and (HERE / f'results/FREEZE_{grp}.json').exists()
        for me in ms:
            for k in range(4):
                out.append((f'test_{me}_s{k}', ok_test, lambda me=me, k=k: sub_test(me, k), True))
    out.append(('dpo_val', is_done('dpo_text') and is_done('dpo_multi'), lambda: sub('src/evaluate.py', 'dpo_val', '--model', 'nemotron'), True))
    if pre and not is_done('preflight'):
        out = [t for t in out if t[0] == 'preflight']
    only = os.environ.get('I3_ONLY') or ((LOCKS / 'only.txt').read_text().strip() if (LOCKS / 'only.txt').exists() else '')
    skip = os.environ.get('I3_SKIP', '')
    if only:
        out = [t for t in out if re.search(only, t[0])]
    if skip:
        out = [t for t in out if not re.search(skip, t[0])]
    return out


def sub_test(method, shard):
    os.environ['ALLOW_TEST'] = 'yash-approved'   # only reached when the approval marker exists
    sub('src/run_test.py', method, str(shard))


def dpo_ok():
    f = G / 'dpo_check.json'
    return f.exists() and json.loads(f.read_text()).get('version') == 3 and json.loads(f.read_text())['resume_ok']


def try_lock(t):
    p = LOCKS / f'{t}.lock'
    try:
        if p.exists() and time.time() - p.stat().st_mtime > 300:
            p.rename(LOCKS / f'{t}.stale.{int(time.time())}')     # owner died: heartbeat older than 5 min
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, os.environ.get('SLURM_JOB_ID', '?').encode()); os.close(fd)
        return p
    except (FileExistsError, FileNotFoundError):
        return None


def main():
    idle_since = None
    while time.time() < DEADLINE - 60:
        transitions()
        if (HERE / 'results/STOP.md').exists():
            print('STOP.md present, exiting', flush=True); return
        ran = False
        for name, ready, body, own in tasks():
            if not ready or is_done(name):
                continue
            if own and STATE['model'] is not None:      # free the GPUs: restart this driver without a model
                print('re-exec for', name, flush=True); os.execv(sys.executable, [sys.executable, *sys.argv])
            p = try_lock(name)
            if p is None:
                continue
            beat = threading.Event()
            threading.Thread(target=lambda: [p.exists() and os.utime(p) for _ in iter(lambda: beat.wait(60), True)], daemon=True).start()
            print(f'=== {name} (job {os.environ.get("SLURM_JOB_ID")}) {time.strftime("%H:%M")}', flush=True)
            code = 0
            try:
                body()
            except SystemExit as e:
                code = e.code or 0
            except Exception:
                traceback.print_exc(); code = 1
            beat.set(); p.unlink(missing_ok=True)
            print(f'=== {name} exit {code} {time.strftime("%H:%M")}', flush=True)
            if code == 0:
                done_file(name).touch()
            elif code == 75:
                return
            else:
                fails = LOCKS / f'{name}.fails'; n = int(fails.read_text() or 0) + 1 if fails.exists() else 1; fails.write_text(str(n))
                if n >= 2:
                    stop(f'{name} failed twice (job {os.environ.get("SLURM_JOB_ID")}), see its log')
                return
            ran = True
            break
        if not ran:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 240:          # nothing to do: do not burn SU waiting
                print('no runnable task, exiting', flush=True); return
            time.sleep(20)
        else:
            idle_since = None


def check():
    """Login node: run the CPU transitions, print how many tasks are runnable and unlocked."""
    transitions()
    if (HERE / 'results/STOP.md').exists():
        print('RUNNABLE STOP'); return
    live = lambda t: (LOCKS / f'{t}.lock').exists() and time.time() - (LOCKS / f'{t}.lock').stat().st_mtime < 300
    r = [n for n, ready, _, _ in tasks() if ready and not is_done(n) and not live(n)]
    left = [n for n, _, _, _ in tasks() if not is_done(n)]
    print('RUNNABLE', len(r), ','.join(r[:6]), 'LEFT', len(left))


if __name__ == '__main__':
    check() if '--check' in sys.argv else main()
