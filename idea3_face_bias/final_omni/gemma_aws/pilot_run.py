"""Run the two pilots (PILOT_PLAN.md) on an AWS GPU instance (g6e.xlarge or larger), with the same tested helper as the Gemma
speech run (gs_colab.py); only the stage script and results folder change. Resumable: run it again and it continues.
Reuses the environment and model download of aws_run.py if they exist on this instance. Results and a zip: ~/idea3_gemma_speech/.

Usage on the instance (inside tmux):   python3 pilot_run.py
"""
import hashlib, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HOME = Path.home()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_colab as q

BASE = HOME / 'idea3_gemma_speech'
q.SCRIPT, q.SUB = 'src/gemma_pilot.py', 'results/gemma_pilot'
q.DRIVE = BASE / 'state_pilot'; q.STATE = q.DRIVE / 'results'
q.ROOT = BASE / 'gs'; q.CODE = q.ROOT / 'final_omni'; q.VENV = BASE / 'gsenv'; q.PY = q.VENV / 'bin/python'
q.HF = BASE / 'hf'; q.P777 = q.ROOT / 'p777'
BUNDLE = HERE / q.BUNDLE


def main():
    q.DRIVE.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    mark = q.ROOT / '.bundle_sha'
    if q.ROOT.exists() and (not mark.exists() or mark.read_text() != sha):
        # a different (older) bundle was unpacked here: unpack this one over it (code and cues), keep the environment and model
        q.ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', 'xzf', str(BUNDLE), '-C', str(q.ROOT)], check=True)
        (q.ROOT / '.unpacked').touch()
        print('unpacked the pilot bundle over the earlier one', flush=True)
    shutil.copy(BUNDLE, q.DRIVE / q.BUNDLE)
    if shutil.which('uv') is None:
        subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)
        os.environ['PATH'] = str(HOME / '.local/bin') + ':' + os.environ['PATH']
    _sh = q.sh
    q.sh = lambda cmd, **k: None if cmd == 'pip -q install uv' else _sh(cmd, **k)
    q.gpu_check()
    q.setup()
    mark.write_text(sha)
    q.run(['perc', 'gate', 'rating', 'report'], hours=4)
    q.sync()
    G = q.CODE / q.SUB
    z = BASE / f'pilot_results_{time.strftime("%m%d_%H%M")}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(G.rglob('*')):
            if p.is_file():
                f.write(p, p.relative_to(G))
    print('RESULTS ZIP:', z, round(z.stat().st_size / 1e6, 1), 'MB')
    print((G / 'report.md').read_text() if (G / 'report.md').exists() else 'no report (a stage stopped early; run again)')


if __name__ == '__main__':
    main()
