"""Submit one resumable stage, following Yash's overnight queue rules (2026-11-24, about 01:20 ET):
 1. batch (gpuA100x4 / gpuH200x8) unless Slurm estimates a wait above 1 h on both (2 h for DPO); else interactive.
 2. Slurm billing (interactive partitions already bill 2x) counts toward the 100 SU cap; worst case is reserved.
 3. interactive walltime <= 1 h; every stage resumes from its rows and saves as it goes.
 4. DPO: batch if it can start within about 2 h.
Account is always bjuv-delta-gpu. Every decision is appended to OVERNIGHT_LOG.md.
Usage: python3 slurm/submit.py NAME "python command" [--hours 0.95] [--dpo] [--debug] [--dry]
"""
import argparse, datetime as dt, json, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
W = Path('/work/hdd/bjuv/ylothe/idea3_final')
ACCOUNT, CAP = 'bjuv-delta-gpu', 150.0   # Yash approved 200 total on Nov 24; ask Yash at 150 before raising
LEDGER = HERE / 'results/su_ledger.jsonl'
# partition: (gpus, cpus, mem, SU per hour for that request)
SPEC = {'gpuA100x4': (2, 16, '110g', 2.0), 'gpuH200x8': (1, 8, '110g', 3.0),
        'gpuA100x4-interactive': (2, 16, '110g', 4.0),
        'gpuA100x8-interactive': (2, 16, '110g', 6.0),   # added with Yash's OK, Nov 24: 3 SU per GPU-hour
        'gpuH200x8-interactive': (1, 8, '110g', 6.0),
        'gpuA40x4-interactive': (2, 16, '110g', 2.0)}   # back to 2x A40, one copy: the 4-GPU 57-min job never backfilled


def run(*a):
    r = subprocess.run(a, capture_output=True, text=True)
    return r.stdout + r.stderr


def su_used():
    """(finished + running so far, worst-case remainder of running/pending) for jobs in our ledger."""
    used = reserve = 0.0
    if not LEDGER.exists():
        return 0.0, 0.0
    for j in map(json.loads, LEDGER.read_text().splitlines()):
        out = run('sacct', '-j', j['job'], '-X', '-n', '-P', '-o', 'ElapsedRaw,State').strip()
        el, st = (out.splitlines()[0].split('|') + [''])[:2] if out else ('0', 'PENDING')
        h = int(el or 0) / 3600
        used += h * j['rate']
        if st.startswith(('RUNNING', 'PENDING')):
            reserve += max(j['hours'] - h, 0) * j['rate']
    return used, reserve


def hms(hours):
    s = int(hours * 3600)
    return f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'


def args_for(part, hours):
    g, c, m, _ = SPEC[part]
    return [f'--account={ACCOUNT}', f'--partition={part}', '--nodes=1', f'--gpus-per-node={g}', '--ntasks=1',
            f'--cpus-per-task={c}', f'--mem={m}', f'--time={hms(hours)}']


def wait_hours(part, hours):
    out = run('sbatch', '--test-only', *args_for(part, hours), '--wrap', 'hostname')
    m = re.search(r'start at (\S+)', out)
    return (dt.datetime.fromisoformat(m.group(1)) - dt.datetime.now()).total_seconds() / 3600 if m else 1e9


def launch(cmd, procs):
    if procs == 1:
        return cmd + '\n'
    # copies on separate GPU pairs; the second starts 90 s later so they do not race on the shared module cache
    return ('set +e\n' + ''.join(f'(sleep {90 * k}; CUDA_VISIBLE_DEVICES={2 * k},{2 * k + 1} {cmd}) &\n' for k in range(procs)) + 'wait\n')


def log(msg):
    with open(HERE / 'OVERNIGHT_LOG.md', 'a') as fh:
        fh.write(f"- {dt.datetime.now():%m-%d %H:%M} CT | {msg}\n")
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('cmd')
    ap.add_argument('--hours', type=float, default=0.417, help='walltime; interactive is capped at 0.95 h')
    ap.add_argument('--dpo', action='store_true'); ap.add_argument('--debug', action='store_true')
    ap.add_argument('--dry', action='store_true'); ap.add_argument('--procs', type=int, default=1); ap.add_argument('--part', default='', help='force one interactive partition (still under the batch-wait rule)')
    a = ap.parse_args()
    waits = {p: wait_hours(p, a.hours) for p in ('gpuA100x4', 'gpuH200x8')}
    best = min(waits, key=waits.get)
    limit = 2.0 if a.dpo else 1.0
    if not a.debug and waits[best] <= limit:
        part, hours = best, a.hours
    else:
        hours = min(a.hours, 0.95)   # interactive limit is 1 h
        iw = {p: wait_hours(p, hours) for p in SPEC if p.endswith('-interactive')}
        # cheaper A40s unless the A100s would start more than 30 min sooner
        part = a.part or 'gpuA40x4-interactive' if a.part or iw['gpuA40x4-interactive'] <= iw['gpuA100x4-interactive'] + 0.5 else 'gpuA100x4-interactive'
    rate = SPEC[part][3]
    used, reserve = su_used()
    need = hours * rate
    desc = f"{a.name}: batch wait est A100x4 {waits['gpuA100x4']:.1f} h, H200 {waits['gpuH200x8']:.1f} h -> {part}, " \
           f"{hours:.2f} h at {rate} SU/h (max {need:.1f}); SU used {used:.1f}, reserved {reserve:.1f}"
    if used + reserve + need > CAP:
        log('REFUSED (100 SU cap): ' + desc); sys.exit(3)
    if a.dry:
        print('DRY', desc); return
    sh = W / 'logs' / f'{a.name}.sbatch'
    sh.write_text('#!/bin/bash\nset -eo pipefail\n'
                  f'source {HERE}/slurm/env.sh\ncd {HERE}\nnvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader\n' + launch(a.cmd, a.procs))
    out = run('sbatch', '--parsable', *args_for(part, hours), f'--job-name=i3_{a.name}', f'--output={W}/logs/{a.name}_%j.log', str(sh))
    m = re.search(r'^(\d+)', out.strip(), re.M)
    if not m:
        log(f'SBATCH FAILED: {desc}: {out.strip()[:300]}'); sys.exit(4)
    with open(LEDGER, 'a') as fh:
        fh.write(json.dumps({'job': m.group(1), 'name': a.name, 'partition': part, 'hours': hours, 'rate': rate,
                             'submitted': dt.datetime.now().isoformat()[:19]}) + '\n')
    log(f'JOB {m.group(1)} ' + desc)


if __name__ == '__main__':
    main()
