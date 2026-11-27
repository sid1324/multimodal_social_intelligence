"""Qwen3-Omni-30B-A3B hearing test and pilot (Nov 27): does a model built for speech hear the spoken remarks, and does it show the
same looks-vs-record pattern as Gemma, typed and spoken? Same code, cues, scenes and rules as the Gemma pilot (PILOT_PLAN.md).
Needs one GPU with at least 80 GB (AWS g7e.2xlarge, Colab G4, A100/H100 80 GB). Resumable: run it again and it continues.

Decision rule, fixed before results (Nov 27): if Qwen3-Omni repeats the spoken remark with the right meaning in at least 90% of the
30 hearing-check scenes, for both voices, it becomes the main model (text vs speech) and Gemma the second-model replication.

Usage (inside tmux):   python3 qwen_run.py        Results and a zip: ~/idea3_gemma_speech/
"""
import hashlib, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HOME = Path.home()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_colab as q

os.environ['GS_MODEL'] = 'qwen'
BASE = HOME / 'idea3_gemma_speech'
q.MODEL_ID, q.REV = 'Qwen/Qwen3-Omni-30B-A3B-Instruct', '26291f793822fb6be9555850f06dfe95f2d7e695'
q.DRIVE = BASE / 'state_qwen'; q.STATE = q.DRIVE / 'results'
q.ROOT = BASE / 'gs'; q.CODE = q.ROOT / 'final_omni'; q.VENV = BASE / 'gsenv'; q.PY = q.VENV / 'bin/python'
q.HF = BASE / 'hf'; q.P777 = q.ROOT / 'p777'
BUNDLE = HERE / q.BUNDLE


def main():
    q.DRIVE.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    mark = q.ROOT / '.bundle_sha'
    if q.ROOT.exists() and (not mark.exists() or mark.read_text() != sha):
        q.ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', 'xzf', str(BUNDLE), '-C', str(q.ROOT)], check=True)
        (q.ROOT / '.unpacked').touch()
        print('unpacked this bundle over an earlier one (environment and models kept)', flush=True)
    shutil.copy(BUNDLE, q.DRIVE / q.BUNDLE)
    if shutil.which('uv') is None:
        subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)
        os.environ['PATH'] = str(HOME / '.local/bin') + ':' + os.environ['PATH']
    _sh = q.sh
    q.sh = lambda cmd, **k: None if cmd == 'pip -q install uv' else _sh(cmd, **k)
    q.gpu_check(min_gb=75)
    q.setup()
    mark.write_text(sha)
    # 1. model check (letters, greedy answer, steering hook), as for Gemma
    q.SCRIPT, q.SUB = 'src/gemma_speech.py', 'results/qwen_speech'
    q.STATE = q.DRIVE / 'results_check'
    q.run(['check'], hours=1)
    if (q.CODE / q.SUB / 'STOP.md').exists():
        print('MODEL CHECK FAILED:', (q.CODE / q.SUB / 'STOP.md').read_text()); return
    # 2. pilot: hearing check, source and implicit cues (typed and spoken), ratings, report
    q.SCRIPT, q.SUB = 'src/gemma_pilot.py', 'results/qwen_pilot'
    q.STATE = q.DRIVE / 'results'
    q.run(['perc', 'gate', 'rating', 'report'], hours=6)
    q.sync()
    z = BASE / f'qwen_results_{time.strftime("%m%d_%H%M")}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for sub in ('results/qwen_speech', 'results/qwen_pilot'):
            G = q.CODE / sub
            for p in sorted(G.rglob('*')):
                if p.is_file():
                    f.write(p, Path(sub.split('/')[1]) / p.relative_to(G))
    print('RESULTS ZIP:', z, round(z.stat().st_size / 1e6, 1), 'MB')
    rep = q.CODE / 'results/qwen_pilot/report.md'
    print(rep.read_text() if rep.exists() else 'no report yet (a stage stopped early; run again)')


if __name__ == '__main__':
    main()
