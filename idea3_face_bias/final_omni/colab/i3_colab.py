"""Colab helper for Idea 3 (plain Python, standard library only, runs in Colab's own Python). The project itself runs in a
separate uv environment with the exact package versions used on Delta. Everything is resumable: results, locks and
adapters are copied to Drive every 2 minutes and restored on start. Test-split tasks stay locked (no approval marker)."""
import json, os, re, shutil, subprocess, sys, tarfile, threading, time
from pathlib import Path
BUNDLE = 'idea3_bundle.tar.gz'
DRIVE = Path('/content/drive/MyDrive/idea3_final') if Path('/content/drive/MyDrive').exists() else Path('/content/idea3_state')
STATE = DRIVE / 'state'
ROOT = Path('/content/i3')
CODE, P777, HF = ROOT / 'final_omni', ROOT / 'p777', ROOT / 'hf'
VENV = Path('/content/i3env')
PY = VENV / 'bin/python'
WEIGHTS = ('nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16', 'e5e9932441de940c9a62185c870ea5bcd4cd24e2')
SYNC = [('results', CODE / 'results'), ('locks', P777 / 'locks'), ('adapters', P777 / 'adapters'), ('logs', P777 / 'logs')]


def sh(cmd, **k):
    print('$', cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **k)


def env(end=None):
    e = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'MPLBACKEND', 'ALLOW_TEST')}
    e.update(P777=str(P777), DATA_DIR=str(P777 / 'data/egonormia'), REF_DIR=str(P777 / 'data/ref'), CACHE_DIR=str(P777 / 'cache'),
             HF_HOME=str(HF), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_MODULES_CACHE='/content/hf_modules',
             TRITON_CACHE_DIR='/content/triton', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
             I3_PREFLIGHT='1', I3_SKIP='^(g2L1|blind)', PATH=f'{VENV}/bin:' + os.environ['PATH'])
    if end:
        e['SLURM_JOB_END_TIME'] = str(int(end))       # the runners stop cleanly 3 minutes before this
        e['SLURM_JOB_ID'] = 'colab'
    return e


def gpu_check(min_gb=75):
    out = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits', shell=True, capture_output=True, text=True).stdout.strip()
    print('GPU:', out or 'NONE')
    assert out, 'No GPU: Runtime > Change runtime type > A100 GPU with High RAM, then run again.'
    mb = int(out.splitlines()[0].split(',')[1])
    assert mb > min_gb * 1000, f'This GPU has {mb/1000:.0f} GB, the model needs about 71 GB. Choose A100 with the High RAM switch on (80 GB) or G4 (96 GB). You are not being charged for anything useful here: disconnect the runtime.'


def copy_new(src, dst):
    n = 0
    for p in src.rglob('*') if src.exists() else []:
        if p.is_dir() or p.name.endswith('.lock') or '.stale.' in p.name or p.suffix == '.tmp':
            continue
        q = dst / p.relative_to(src)
        s = p.stat()
        if q.exists() and q.stat().st_size == s.st_size and abs(q.stat().st_mtime - s.st_mtime) < 2:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        t = q.with_name(q.name + '.part')
        shutil.copy2(p, t); os.replace(t, q); os.utime(q, (s.st_mtime, s.st_mtime)); n += 1
    return n


def sync():
    return sum(copy_new(local, STATE / name) for name, local in SYNC)


def setup():
    if not str(DRIVE).startswith('/content/drive'):
        print('WARNING: Google Drive is not mounted, so progress is LOST if this machine disconnects. Mount it first: Command Palette > "Colab: Mount Google Drive to Server", then restart from the top.')
    DRIVE.mkdir(parents=True, exist_ok=True)
    if not (ROOT / '.unpacked').exists():
        found = [c for c in (Path('/content') / BUNDLE, DRIVE / BUNDLE) if c.exists()]
        assert found, f'Upload {BUNDLE} first: right-click it in the VS Code explorer > "Upload to Colab" (goes to /content).'
        ROOT.mkdir(parents=True, exist_ok=True)
        sh(['tar', 'xzf', str(found[0]), '-C', str(ROOT)])
        if not (DRIVE / BUNDLE).exists():
            shutil.copy(found[0], DRIVE / BUNDLE)             # so a later machine can start without another upload
        (ROOT / '.unpacked').touch()
        for name, local in SYNC:                               # restore earlier progress from Drive
            if (STATE / name).exists():
                print('restored', name, copy_new(STATE / name, local), 'files')
    for d in ('locks', 'adapters', 'logs', 'cache'):
        (P777 / d).mkdir(parents=True, exist_ok=True)
    if not (VENV / '.ok').exists():
        sh('pip -q install uv')
        sh(f'uv venv -q --clear --python 3.12.16 {VENV} || uv venv -q --clear --python 3.12 {VENV}')
        sh(f'uv pip install -q --python {PY} -r {CODE}/colab/requirements.lock --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128')
        (VENV / '.ok').touch()
    if not (P777 / 'data/.images_ok').exists():
        # public EgoNormia images, pinned revision, every file checked against its recorded SHA-256 (as at midterm)
        sh([str(PY), '-I', 'src/download_data.py', '--workers', '32'], cwd=CODE, env=env())
        rep = json.loads((P777 / 'data/egonormia/verify_report.json').read_text())
        assert rep['raw_sha_ok'] == 1853 and rep['preview_sha_ok'] == 1853, {k: rep[k] for k in ('raw_sha_ok', 'preview_sha_ok')}
        (P777 / 'data/.images_ok').touch()
    e = env(); e['HF_HUB_OFFLINE'] = '0'; e['TRANSFORMERS_OFFLINE'] = '0'
    if not (HF / '.kernels_ok').exists():
        # pinned prebuilt Mamba GPU kernels (public), the same revisions as on Delta
        sh([str(PY), '-c', "from kernels.utils import install_kernel as f; [print(f(r, revision=v)) for r, v in (('kernels-community/mamba-ssm', '768416abf0ad7262a23d83c6efef10b9a4786489'), ('kernels-community/causal-conv1d', '6e9a58272220a363dc50e59cba737abcedd7b59f'))]"], env=e)
        (HF / '.kernels_ok').touch()
    if not (HF / '.weights_ok').exists():
        print('Downloading the model (62 GB), about 5 to 15 minutes', flush=True)
        code = f"from huggingface_hub import snapshot_download; print(snapshot_download('{WEIGHTS[0]}', revision='{WEIGHTS[1]}', ignore_patterns=['media/*'], max_workers=16))"
        sh([str(PY), '-c', code], env=e); (HF / '.weights_ok').touch()
    out = subprocess.run([str(PY), '-c', "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))"], env=env(), capture_output=True, text=True)
    print('environment:', out.stdout.strip() or out.stderr[-800:])
    assert out.returncode == 0 and 'torch' not in out.stderr[:0]
    print('SETUP DONE')


def check(end=None):
    out = subprocess.run([str(PY), 'src/driver.py', '--check'], cwd=CODE, env=env(end), capture_output=True, text=True)
    m = [l for l in out.stdout.splitlines() if l.startswith('RUNNABLE')]
    return m[-1] if m else 'CHECK FAILED ' + out.stderr[-500:]


def analyses(end):
    """CPU summaries, each run once its inputs are complete. Never touches the test split."""
    L = P777 / 'locks'; done = lambda n: (L / f'{n}.done').exists()
    jobs = []
    if done('blind_blank') and done('blind_wrong'):
        jobs.append(('a_blind', [['src/analyze.py', 'blind']]))
    sw = [f'sweep_base'] + [f'sweep_naive_{n}_a{a}' for n in ('L13', 'L21', 'L29', 'L36', 'L21-51') for a in (0.5, 1.0)]
    if all(done(s) for s in sw):
        jobs.append(('a_sweep', [['src/analyze.py', 'sweep'], ['src/freeze.py', 'A']]))
    if done('dpo_val'):
        jobs.append(('a_dpo', [['src/analyze.py', 'dpo']]))
    for name, cmds in jobs:
        if not (L / f'{name}.done').exists():
            ok = True
            for c in cmds:
                r = subprocess.run([str(PY), *c], cwd=CODE, env=env(end), capture_output=True, text=True)
                print(r.stdout[-5000:], r.stderr[-1500:] if r.returncode else '', flush=True); ok &= r.returncode == 0
            if ok:
                (L / f'{name}.done').touch()


def run(max_hours=11.0, auto_stop=True):
    """Run everything that is ready, resumably, for at most max_hours (a spending limit). Then copy to Drive and, if
    auto_stop, release the GPU so it stops using compute units."""
    end = time.time() + max_hours * 3600
    subprocess.run('pkill -f src/driver.py', shell=True)
    subprocess.run('pkill -f src/dpo_', shell=True)
    stop_sync = threading.Event()

    def loop():
        while not stop_sync.wait(120):
            try:
                sync()
            except Exception as ex:
                print('sync problem (will retry):', ex, flush=True)
    threading.Thread(target=loop, daemon=True).start()
    log = P777 / 'logs/colab_driver.log'
    stale = 0
    try:
        while time.time() < end - 300:
            for l in (P777 / 'locks').glob('*.lock'):
                l.unlink()
            before = {p.name for p in (P777 / 'locks').glob('*.done')}, sum(p.stat().st_size for p in (CODE / 'results').rglob('*.jsonl'))
            p = subprocess.Popen([str(PY), 'src/driver.py'], cwd=CODE, env=env(end), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
            try:
                with open(log, 'a') as fh:
                    for line in p.stdout:
                        fh.write(line); fh.flush()
                        if re.search(r'^===|PREFLIGHT|finished|s/pass|DPO_CHECK|GRAD|Traceback|Error|STOP|deadline|\'step\'|DPO DONE|no runnable|reference done|pairs', line):
                            print(time.strftime('%H:%M'), line.rstrip()[:400], flush=True)
                p.wait()
            finally:
                if p.poll() is None:
                    p.kill(); p.wait()
            analyses(end); sync()
            c = check(end); print(time.strftime('%H:%M'), c, flush=True)
            if (CODE / 'results/STOP.md').exists():
                print('STOPPED:', (CODE / 'results/STOP.md').read_text()); break
            m = re.match(r'RUNNABLE (\d+)', c)
            if not m or int(m.group(1)) == 0:
                print('Nothing left that is allowed to run.'); break
            after = {p.name for p in (P777 / 'locks').glob('*.done')}, sum(p.stat().st_size for p in (CODE / 'results').rglob('*.jsonl'))
            stale = stale + 1 if after == before else 0
            if stale >= 3:
                print('Three rounds without progress, stopping. See', log); break
    finally:
        stop_sync.set(); time.sleep(1)
        for _ in range(3):
            if sync() == 0:
                break
        print('All progress copied to Drive:', STATE)
    status()
    pack()
    if auto_stop:
        print('Releasing the GPU so it stops using compute units. Re-run the notebook to continue.')
        time.sleep(45)                                    # let Drive finish uploading
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as ex:
            print('Could not release the GPU automatically. Do it by hand NOW: Command Palette > "Colab: Remove Server".', ex)


def status():
    L = P777 / 'locks'
    print('done:', ', '.join(sorted(p.stem for p in L.glob('*.done'))))
    for f in ('gate/preflight.json', 'gate/blind_nemotron_summary.json', 'gate/dpo_check.json'):
        q = CODE / 'results' / f
        if q.exists():
            print(f, json.dumps(json.loads(q.read_text()))[:900])
    q = CODE / 'results/FREEZE_A.md'
    if q.exists():
        print(q.read_text())


def pack():
    out = DRIVE / f"idea3_results_{time.strftime('%m%d_%H%M')}.tar.gz"
    with tarfile.open('/content/res.tar.gz', 'w:gz') as t:
        t.add(CODE / 'results', 'results')
        for n in ('locks', 'logs'):
            t.add(P777 / n, n)
    shutil.copy('/content/res.tar.gz', out); print('Results archive:', out)
