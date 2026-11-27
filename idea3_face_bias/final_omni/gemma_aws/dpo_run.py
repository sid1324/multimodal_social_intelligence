"""Run DPO on Gemma 4 12B (src/gemma_dpo.py) on one GPU with at least 40 GB (Colab G4/A100, AWS g6e.xlarge, lab server), with the
same tested helper as the Gemma speech run (gs_colab.py). Resumable: run it again and it continues where it stopped.
Order: rows (about 15 min), pairs, DPO-text (about 1 to 2 h), DPO-multi (about 2 to 4 h), evaluation (about 1 h), report.
Results, adapters and a zip: ~/idea3_gemma_speech/ (results/gemma_dpo inside). Nothing here touches the test split.

Usage (inside tmux or a Colab terminal):   python3 dpo_run.py
"""
import hashlib, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HOME = Path.home()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_colab as q

BASE = HOME / 'idea3_gemma_speech'
q.SCRIPT, q.SUB = 'src/gemma_dpo.py', 'results/gemma_dpo'
q.DRIVE = BASE / 'state_dpo'; q.STATE = q.DRIVE / 'results'
q.ROOT = BASE / 'gs'; q.CODE = q.ROOT / 'final_omni'; q.VENV = BASE / 'gsenv'; q.PY = q.VENV / 'bin/python'
q.HF = BASE / 'hf'; q.P777 = q.ROOT / 'p777'
BUNDLE = HERE / q.BUNDLE
PEFT = 'peft==0.21.2'


def main():
    q.DRIVE.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    mark = q.ROOT / '.bundle_sha'
    if q.ROOT.exists() and (not mark.exists() or mark.read_text() != sha):
        q.ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', 'xzf', str(BUNDLE), '-C', str(q.ROOT)], check=True)
        (q.ROOT / '.unpacked').touch()
        print('unpacked this bundle over an earlier one (environment and model kept)', flush=True)
    shutil.copy(BUNDLE, q.DRIVE / q.BUNDLE)
    if shutil.which('uv') is None:
        subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)
        os.environ['PATH'] = str(HOME / '.local/bin') + ':' + os.environ['PATH']
    _sh = q.sh
    q.sh = lambda cmd, **k: None if cmd == 'pip -q install uv' else _sh(cmd, **k)
    q.gpu_check()
    q.setup()
    if subprocess.run([str(q.PY), '-c', 'import peft'], capture_output=True).returncode:
        q.sh(f'uv pip install -q --python {q.PY} {PEFT}')
    mark.write_text(sha)
    assert (q.CODE / 'results/gemma/g2_rows.jsonl').exists(), 'the bundle must contain results/gemma (base answers from the speech run)'
    q.run(['rows', 'pairs', 'train text', 'train multi', 'eval', 'report'], hours=12)
    q.sync()
    G = q.CODE / q.SUB
    z = BASE / f'dpo_results_{time.strftime("%m%d_%H%M")}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(G.rglob('*')):
            if p.is_file() and p.name not in ('last.pt', 'ref.json'):   # adapters are included; optimizer state is not
                f.write(p, p.relative_to(G))
    print('RESULTS ZIP:', z, round(z.stat().st_size / 1e6, 1), 'MB')
    print((G / 'report.md').read_text() if (G / 'report.md').exists() else 'no report yet (a stage stopped early; run again)')


if __name__ == '__main__':
    main()
