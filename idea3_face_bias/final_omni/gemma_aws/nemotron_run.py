"""Nemotron 3 Nano Omni 30B hearing test and pilot (Nov 27): the same model check, hearing check, source and implicit cues (typed and
spoken) and ratings as the Gemma and Qwen3-Omni pilots (PILOT_PLAN.md). Nemotron runs in its own pinned environment
(colab/requirements.lock: torch 2.9.1+cu128, transformers 5.6.2, pinned Mamba kernels), as on Delta.
Needs one GPU with at least 80 GB (AWS g7e.2xlarge, Colab G4). Resumable: run it again and it continues.

Risk, checked first: Nemotron's Mamba layers use prebuilt CUDA kernels that were only ever run on A40, A100 and H200. The model check
(about 10 minutes after the download) is the first forward pass; if the kernels do not support this GPU it fails there, early.

Decision rule (Nov 27, before results): the model that repeats the spoken remark with the right meaning in at least 90% of the 30
hearing-check scenes, for both voices, becomes the main model; if both Nemotron and Qwen3-Omni do, Nemotron (most existing work).

Usage (inside tmux):   python3 nemotron_run.py        Results and a zip: ~/idea3_gemma_speech/
"""
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HOME = Path.home()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_colab as q

os.environ['GS_MODEL'] = 'nemotron'
BASE = HOME / 'idea3_gemma_speech'
WEIGHTS = ('nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16', 'e5e9932441de940c9a62185c870ea5bcd4cd24e2')
KERNELS = (('kernels-community/mamba-ssm', '768416abf0ad7262a23d83c6efef10b9a4786489'),
           ('kernels-community/causal-conv1d', '6e9a58272220a363dc50e59cba737abcedd7b59f'))
q.DRIVE = BASE / 'state_nemotron'; q.STATE = q.DRIVE / 'results'
q.ROOT = BASE / 'gs'; q.CODE = q.ROOT / 'final_omni'; q.VENV = BASE / 'nemenv'; q.PY = q.VENV / 'bin/python'
q.HF = BASE / 'hf'; q.P777 = q.ROOT / 'p777'
BUNDLE = HERE / q.BUNDLE


def setup():
    q.DRIVE.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    mark = q.ROOT / '.bundle_sha'
    if not mark.exists() or mark.read_text() != sha:
        q.ROOT.mkdir(parents=True, exist_ok=True)
        q.sh(['tar', 'xzf', str(BUNDLE), '-C', str(q.ROOT)])
        (q.ROOT / '.unpacked').touch(); mark.write_text(sha)
    if q.STATE.exists():
        print('restored from earlier state:', q.copy_new(q.STATE, q.CODE / q.SUB), 'files')
    if shutil.which('uv') is None:
        q.sh('curl -LsSf https://astral.sh/uv/install.sh | sh')
        os.environ['PATH'] = str(HOME / '.local/bin') + ':' + os.environ['PATH']
    if not (q.VENV / '.ok').exists():
        q.sh(f'uv venv -q --clear --python 3.12 {q.VENV}')
        q.sh(f'uv pip install -q --python {q.PY} -r {q.CODE}/colab/requirements.lock --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128')
        (q.VENV / '.ok').touch()
    e = q.env()
    for d in ('locks', 'adapters', 'logs', 'cache'):
        (q.P777 / d).mkdir(parents=True, exist_ok=True)
    if not (q.P777 / 'data/.images_ok').exists():
        q.sh([str(q.PY), '-I', 'src/download_data.py', '--workers', '32'], cwd=q.CODE, env=e)
        rep = json.loads((q.P777 / 'data/egonormia/verify_report.json').read_text())
        assert rep['raw_sha_ok'] == 1853 and rep['preview_sha_ok'] == 1853, rep
        (q.P777 / 'data/.images_ok').touch()
    e2 = dict(e, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
    if not (q.HF / '.kernels_ok').exists():
        code = 'from kernels.utils import install_kernel as f; ' + '; '.join(f"print(f('{r}', revision='{v}'))" for r, v in KERNELS)
        q.sh([str(q.PY), '-c', code], env=e2); (q.HF / '.kernels_ok').touch()
    if not (q.HF / '.weights_ok_nemotron').exists():
        print('Downloading Nemotron (about 62 GB)', flush=True)
        q.sh([str(q.PY), '-c', f"from huggingface_hub import snapshot_download as s; print(s('{WEIGHTS[0]}', revision='{WEIGHTS[1]}', ignore_patterns=['media/*'], max_workers=16))"], env=e2)
        (q.HF / '.weights_ok_nemotron').touch()
    out = subprocess.run([str(q.PY), '-c', 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'],
                         env=e, capture_output=True, text=True)
    print('environment:', out.stdout.strip() or out.stderr[-800:], flush=True)
    (q.CODE / q.SUB).mkdir(parents=True, exist_ok=True)
    (q.CODE / q.SUB / 'environment.txt').write_text(out.stdout + out.stderr[-2000:])
    assert out.returncode == 0


def main():
    q.gpu_check(min_gb=75)
    q.SCRIPT, q.SUB = 'src/gemma_speech.py', 'results/nemotron_speech'
    q.STATE = q.DRIVE / 'results_check'
    setup()
    q.run(['check'], hours=1)   # first forward pass: fails here, early, if the Mamba kernels do not support this GPU
    if (q.CODE / q.SUB / 'STOP.md').exists() or not (q.CODE / q.SUB / 'check.json').exists():
        print('MODEL CHECK FAILED OR DID NOT FINISH: see the log above'); return
    q.SCRIPT, q.SUB = 'src/gemma_pilot.py', 'results/nemotron_pilot'
    q.STATE = q.DRIVE / 'results'
    q.run(['perc', 'gate', 'rating', 'report'], hours=8)
    q.sync()
    z = BASE / f'nemotron_results_{time.strftime("%m%d_%H%M")}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for sub in ('results/nemotron_speech', 'results/nemotron_pilot'):
            G = q.CODE / sub
            for p in sorted(G.rglob('*')):
                if p.is_file():
                    f.write(p, Path(sub.split('/')[1]) / p.relative_to(G))
    print('RESULTS ZIP:', z, round(z.stat().st_size / 1e6, 1), 'MB')
    rep = q.CODE / 'results/nemotron_pilot/report.md'
    print(rep.read_text() if rep.exists() else 'no report yet (a stage stopped early; run again)')


if __name__ == '__main__':
    main()
