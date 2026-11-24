"""Watchdog (Nov 24, after jobs sat unnoticed for two hours). Prints one line per NEW problem or milestone; loops every 60 s.
Checks: queue stuck, idle lanes, job failures, tracebacks, stalled logs, STOP.md, failed tasks, keepalive dead, SU level,
finished tasks. Safe automatic fixes only: restart a dead keepalive; cut a job pending over 25 min to 18 min (keeps its place).
Usage: python3 slurm/health.py [END_EPOCH] [--once]"""
import datetime as dt, glob, json, os, re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
W = Path('/work/hdd/bjuv/ylothe/idea3_final')
LANES = ('gpuA40x4-interactive', 'gpuA100x4-interactive', 'gpuH200x8-interactive')
LEDGER = HERE / 'results/su_ledger.jsonl'
seen, last = set(), {}


def run(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ''


def say(key, msg, every=None):
    now = time.time()
    if key in last and (every is None or now - last[key] < every):
        return
    last[key] = now
    print(f"{dt.datetime.now():%H:%M} {msg}", flush=True)


def secs(t):
    d, _, r = t.partition('-') if '-' in t else ('0', '', t)
    p = [int(x) for x in r.split(':')]
    while len(p) < 3:
        p.insert(0, 0)
    return int(d) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def check(end):
    if (HERE / 'results/STOP.md').exists():
        say('stop', 'STOP.md: ' + (HERE / 'results/STOP.md').read_text()[:200])
    if not [l for l in run('ps', '-u', 'ylothe', '-o', 'pid,args').splitlines() if re.search(r'/bin/bash slurm/keepalive2\.sh', l)] and time.time() < end and not (HERE / 'results/STOP.md').exists():
        log = (W / 'logs/keepalive2.log').read_text()[-200:] if (W / 'logs/keepalive2.log').exists() else ''
        if 'ALL DONE' in log:
            say('done', 'ALL DONE: no runnable or pending tasks left')
        else:
            say('ka', 'keepalive was dead, restarting it', 600)
            subprocess.Popen(['setsid', 'nohup', 'slurm/keepalive2.sh', str(end)], cwd=HERE, stdout=open(W / 'logs/keepalive2.log', 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    klog = W / 'logs/keepalive2.log'
    if klog.exists() and time.time() - klog.stat().st_mtime > 600:
        say('klog', f'keepalive log silent for {(time.time() - klog.stat().st_mtime) / 60:.0f} min', 900)
    q = [l.split('|') for l in run('squeue', '-u', 'ylothe', '-h', '-o', '%i|%P|%T|%V|%l|%M|%j').splitlines() if '|' in l]
    mine = [x for x in q if x[6].startswith('i3_')]
    running = [x for x in mine if x[2] == 'RUNNING']
    for jid, p, st, sub, lim, el, _ in mine:
        wait = (dt.datetime.now() - dt.datetime.fromisoformat(sub)).total_seconds() / 60
        if st == 'PENDING' and wait > 25:
            if secs(lim) > 1080:
                run('scontrol', 'update', f'JobId={jid}', 'TimeLimit=00:18:00')
                say(f'cut{jid}', f'{jid} {p} pending {wait:.0f} min: cut to 18 min')
            if not running:
                say(f'wait{jid}', f'QUEUE: {jid} {p} pending {wait:.0f} min, nothing of ours running', 1800)
        if st == 'RUNNING':
            say(f'start{jid}', f'{jid} started on {p}')
            f = W / f'logs/drv_{jid}.log'
            if f.exists() and secs(el) > 600 and time.time() - f.stat().st_mtime > 420:
                say(f'stall{jid}', f'STALL: {jid} log silent {(time.time() - f.stat().st_mtime) / 60:.0f} min', 900)
    run_line = [l for l in klog.read_text().splitlines() if 'RUNNABLE' in l][-1:] if klog.exists() else []
    m = re.search(r'RUNNABLE (\d+)', run_line[0]) if run_line else None
    if m and int(m.group(1)) > 0:
        for p in LANES:
            if not any(x[1] == p for x in mine):
                say(f'idle{p}', f'IDLE LANE: {p} has no job although {m.group(1)} tasks are runnable', 900)
    if LEDGER.exists():
        for j in [json.loads(l) for l in LEDGER.read_text().splitlines()][-8:]:
            s = run('sacct', '-j', j['job'], '-X', '-n', '-P', '-o', 'State').strip().split('\n')[0]
            if s and not s.startswith(('COMPLETED', 'RUNNING', 'PENDING', 'CANCELLED')):
                say(f'bad{j["job"]}', f'JOB {j["job"]} ended {s}')
    for f in sorted((W / 'locks').glob('*.fails')):
        say(f'f{f.name}{f.read_text()}', f'TASK FAILED: {f.stem} count {f.read_text()}')
    for f in sorted((W / 'logs').glob('drv_*.log'), key=os.path.getmtime)[-4:]:
        t = f.read_text(errors='ignore')
        n = t.count('Traceback')
        if n:
            say(f'tb{f.name}{n}', f'TRACEBACK in {f.name}: ' + t[t.rfind('Traceback'):][-300:].replace('\n', ' | '))
    for f in sorted((W / 'locks').glob('*.done')):
        if f.stat().st_mtime > time.time() - 300:
            say(f'd{f.name}', f'DONE {f.stem}')
    try:
        sys.path.insert(0, str(HERE / 'slurm')); import submit
        u, r = submit.su_used()
        for lv in (100, 125, 140):
            if u + r > lv:
                say(f'su{lv}', f'SU used+reserved {u + r:.0f} (limit 150)')
    except Exception as e:
        say('suerr', f'SU check failed: {e}', 1800)


if __name__ == '__main__':
    end = int(next((x for x in sys.argv[1:] if x.isdigit()), 1795636800))
    if '--once' in sys.argv:
        check(end); sys.exit()
    while time.time() < end:
        try:
            check(end)
        except Exception as e:
            print('watchdog error', e, flush=True)
        time.sleep(60)
