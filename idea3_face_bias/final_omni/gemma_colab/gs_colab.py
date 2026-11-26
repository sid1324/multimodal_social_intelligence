"""Colab helper for the Gemma 4 12B speech-transfer test (src/gemma_speech.py). Plain Python. The experiment runs in its own uv
environment (torch 2.9.1 + transformers 5.22.2 + pillow 12.3.0). Resumable: results are copied to Drive every minute and restored on
start. The test split is not in this notebook."""
import json, os, re, shutil, subprocess, threading, time
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive/idea3_gemma_speech') if Path('/content/drive/MyDrive').exists() else Path('/content/gs_state')
STATE, ROOT, CODE, VENV = DRIVE / 'results', Path('/content/gs'), Path('/content/gs/final_omni'), Path('/content/gsenv')
PY, HF, P777 = VENV / 'bin/python', Path('/content/hf'), Path('/content/gs/p777')
BUNDLE = 'gemma_speech_bundle.tar.gz'
REV = '43c9d014b231474fa7212dbb48a762566c2b255d'   # google/gemma-4-12B-it


def sh(cmd, **k):
    print('$', cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), flush=True); subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **k)


def env(stop_at=None):
    e = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'MPLBACKEND', 'ALLOW_TEST')}
    e.update(P777=str(P777), DATA_DIR=str(P777 / 'data/egonormia'), REF_DIR=str(P777 / 'data/ref'), HF_HOME=str(HF), HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', MPLBACKEND='Agg', PATH=f'{VENV}/bin:' + os.environ['PATH'])
    if stop_at:
        e['STOP_AT'] = str(stop_at)
    return e


def gpu_check(min_gb=38):
    out = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits', shell=True, capture_output=True, text=True).stdout.strip()
    print('GPU:', out or 'NONE'); assert out, 'No GPU. Choose A100 and connect again.'
    assert int(out.splitlines()[0].split(',')[1]) > min_gb * 1000, 'GPU is smaller than expected (need about 40 GB: A100, H100 or G4); disconnect so it does not use units.'


def copy_new(src, dst):
    n = 0
    for p in src.rglob('*') if src.exists() else []:
        if p.is_dir() or p.suffix == '.tmp':
            continue
        q = dst / p.relative_to(src); s = p.stat()
        if q.exists() and q.stat().st_size == s.st_size and abs(q.stat().st_mtime - s.st_mtime) < 2:
            continue
        q.parent.mkdir(parents=True, exist_ok=True); t = q.with_name(q.name + '.part'); shutil.copy2(p, t); os.replace(t, q); os.utime(q, (s.st_mtime, s.st_mtime)); n += 1
    return n


def sync():
    return copy_new(CODE / 'results/gemma', STATE)


def setup():
    if not str(DRIVE).startswith('/content/drive'):
        print('WARNING: Drive is not mounted: progress is LOST on disconnect. Command Palette > "Colab: Mount Google Drive to Server", then run from the top.')
    DRIVE.mkdir(parents=True, exist_ok=True)
    if not (ROOT / '.unpacked').exists():
        found = [c for c in (Path('/content') / BUNDLE, DRIVE / BUNDLE) if c.exists()]
        assert found, f'Upload {BUNDLE} first (right-click it in VS Code > "Upload to Colab").'
        ROOT.mkdir(parents=True, exist_ok=True); sh(['tar', 'xzf', str(found[0]), '-C', str(ROOT)])
        if not (DRIVE / BUNDLE).exists():
            shutil.copy(found[0], DRIVE / BUNDLE)
        if STATE.exists():
            print('restored from Drive:', copy_new(STATE, CODE / 'results/gemma'), 'files')
        (ROOT / '.unpacked').touch()
    if not (VENV / '.ok').exists():
        sh('pip -q install uv'); sh(f'uv venv -q --clear --python 3.12 {VENV}')
        sh(f'uv pip install -q --python {PY} --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128 torchvision==0.24.1+cu128 transformers==5.22.2 "accelerate>=1.0" pillow==12.3.0 soundfile pyyaml matplotlib numpy requests huggingface_hub hf_xet')
        (VENV / '.ok').touch()
    e = env()
    if not (P777 / 'data/.images_ok').exists():
        sh([str(PY), '-I', 'src/download_data.py', '--workers', '32'], cwd=CODE, env=e)
        rep = json.loads((P777 / 'data/egonormia/verify_report.json').read_text())
        assert rep['raw_sha_ok'] == 1853 and rep['preview_sha_ok'] == 1853, rep
        (P777 / 'data/.images_ok').touch()
    if not (P777 / 'data/omi/.ok').exists():
        # One Million Impressions faces (public, CC BY-NC-SA 4.0), the midterm's pinned commit; used by the exploratory face arm
        sh(f"git clone -q https://github.com/jcpeterson/omi.git {P777 / 'data/omi'} && cd {P777 / 'data/omi'} && git checkout -q 53bb3c13113afc3ae67aa6ede4ac9dfe33f306af && ls images | wc -l")
        (P777 / 'data/omi/.ok').touch()
    if not (HF / '.weights_ok').exists():
        e2 = dict(env(), HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
        print('Downloading the model (about 24 GB)', flush=True)
        sh([str(PY), '-c', f"from huggingface_hub import snapshot_download as s; print(s('google/gemma-4-12B-it', revision='{REV}', max_workers=16))"], env=e2); (HF / '.weights_ok').touch()
    out = subprocess.run([str(PY), '-c', 'import torch, transformers, PIL; print(torch.__version__, transformers.__version__, PIL.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'], env=env(), capture_output=True, text=True)
    print('environment:', out.stdout.strip() or out.stderr[-600:]); assert out.returncode == 0; print('SETUP DONE')


def stage(name, end):
    log = CODE / 'results/gemma/colab_log.txt'; log.parent.mkdir(parents=True, exist_ok=True); print(f'=== {name}', time.strftime('%H:%M'), flush=True)
    p = subprocess.Popen([str(PY), 'src/gemma_speech.py', *name.split()], cwd=CODE, env=env(end), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    with open(log, 'a') as fh:
        for line in p.stdout:
            fh.write(line); fh.flush()
            if not re.search(r'it/s|Loading checkpoint|Loading weights|FutureWarning|UserWarning|^\s*$', line):
                print(line.rstrip()[:420], flush=True)
    p.wait(); print(f'=== {name} exit {p.returncode}', time.strftime('%H:%M'), flush=True); return p.returncode


def run(stages, hours=8.0):
    """Runs the stages in order, resumably, until done or the time limit. Stops at the first stage that fails or stops itself."""
    end = time.time() + hours * 3600; stop_sync = threading.Event()
    def loop():
        while not stop_sync.wait(60):
            try: sync()
            except Exception as ex: print('sync problem (will retry):', ex, flush=True)
    threading.Thread(target=loop, daemon=True).start()
    try:
        for s in stages:
            code = stage(s, end)
            if (CODE / 'results/gemma/STOP.md').exists():
                print('STOPPED BY A GATE:', (CODE / 'results/gemma/STOP.md').read_text()); break
            if code == 75 or time.time() > end - 120:
                print('time limit reached; run this cell again to continue'); break
            if code:
                print('stage failed; send this output to Claude'); break
        else:
            print('ALL STAGES IN THIS CELL DONE')
    finally:
        stop_sync.set(); time.sleep(1)
        for _ in range(3):
            if sync() == 0: break
        print('copied to Drive:', STATE)


def release():
    print('Releasing the GPU'); time.sleep(30)
    try:
        from google.colab import runtime; runtime.unassign()
    except Exception as ex:
        print('Could not release automatically: Command Palette > "Colab: Remove Server" NOW.', ex)
