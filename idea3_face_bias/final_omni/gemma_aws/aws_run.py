"""Run the Gemma speech-transfer experiment on an AWS GPU instance (g6e.xlarge or larger), with the SAME tested code as the
Colab notebook (gemma_colab/gs_colab.py): only the paths change, from /content and Google Drive to the instance's disk.
Resumable: run it again after a stop and it continues. Results and a zip end up in ~/idea3_gemma_speech/.

Usage on the instance (inside tmux, so a dropped SSH connection does not stop it):
    python3 aws_run.py            # setup, Part A (checks, bias gate), Part B (steering, transfer, faces, report), zip
"""
import os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HOME = Path.home()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gs_colab as q

BASE = HOME / 'idea3_gemma_speech'
q.DRIVE = BASE / 'state'; q.STATE = q.DRIVE / 'results'           # plays the role of the Drive folder (survives restarts)
q.ROOT = BASE / 'gs'; q.CODE = q.ROOT / 'final_omni'; q.VENV = BASE / 'gsenv'; q.PY = q.VENV / 'bin/python'
q.HF = BASE / 'hf'; q.P777 = q.ROOT / 'p777'
BUNDLE = Path(__file__).resolve().parent / q.BUNDLE


def main():
    q.DRIVE.mkdir(parents=True, exist_ok=True)
    if not (q.DRIVE / q.BUNDLE).exists():
        shutil.copy(BUNDLE, q.DRIVE / q.BUNDLE)
    if shutil.which('uv') is None:
        subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)   # official uv installer
        os.environ['PATH'] = str(HOME / '.local/bin') + ':' + os.environ['PATH']
    _sh = q.sh   # Colab's 'pip install uv' is replaced by the uv installed above (Ubuntu's system pip refuses global installs)
    q.sh = lambda cmd, **k: None if cmd == 'pip -q install uv' else _sh(cmd, **k)
    q.gpu_check()
    q.setup()
    G = q.CODE / 'results/gemma'
    q.run(['check', 'perc', 'g2', 'gate'], hours=5)
    if (G / 'STOP.md').exists():
        print('Stopped in Part A:', (G / 'STOP.md').read_text())
    elif not (G / 'gate.json').exists():
        print('Part A not finished: run aws_run.py again to continue.'); return
    else:
        q.run(['directions', 'sweep', 'select', 'transfer', 'report', 'faces', 'report'], hours=9)
    q.sync()
    z = BASE / f'gemma_results_{time.strftime("%m%d_%H%M")}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(G.rglob('*')):
            if p.is_file() and 'acts' not in p.parts:
                f.write(p, p.relative_to(G))
    print('RESULTS ZIP:', z, round(z.stat().st_size / 1e6, 1), 'MB')
    print('Copy it home with:  scp -i <your-key.pem> ubuntu@<instance-ip>:' + str(z) + ' .   then STOP the instance in the console.')


if __name__ == '__main__':
    main()
