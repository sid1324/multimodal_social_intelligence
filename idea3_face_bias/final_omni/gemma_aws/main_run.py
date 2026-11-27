"""One runner for the final-study jobs, any model, any Linux GPU machine (AWS g6e/g7e, Colab G4/A100, lab server).
Same tested helper as before (gs_colab.py); resumable: run the same command again and it continues where it stopped.

  python3 main_run.py --model qwen  --job steering   # direction from typed remarks, sweep (L1, L2), speech transfer with controls, faces
  python3 main_run.py --model qwen  --job dpo        # DPO-text and DPO-multi (3 epochs), evaluation, report
  python3 main_run.py --model gemma --job fixeval    # every fix on looks vs record claims, behavior, floor, accuracy (validation)
  python3 main_run.py --model qwen  --job fixeval    # needs this machine to hold the Qwen steering and DPO results (see RUN_FINAL.md)

GPU: Gemma 40 GB or more; Qwen3-Omni 80 GB or more. Results and a zip: ~/idea3_gemma_speech/. Nothing here runs the test split.
"""
import argparse, hashlib, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HOME = Path.home()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_colab as q

MODELS = {'gemma': ('google/gemma-4-12B-it', '43c9d014b231474fa7212dbb48a762566c2b255d', 38),
          'qwen': ('Qwen/Qwen3-Omni-30B-A3B-Instruct', '26291f793822fb6be9555850f06dfe95f2d7e695', 75)}
JOBS = {'steering': ('src/gemma_speech.py', ['check', 'perc', 'g2', 'gate', 'directions', 'sweep', 'select', 'transfer', 'report', 'faces', 'report'], 24),
        'dpo': ('src/gemma_dpo.py', ['rows', 'pairs', 'train text', 'train multi', 'eval', 'report'], 24),
        'fixeval': ('src/fix_eval.py', ['run', 'report val'], 12)}
PEFT = 'peft==0.21.2'


def sub_for(model, job):
    if job == 'steering':
        return 'results/gemma' if model == 'gemma' else f'results/{model}_speech'
    return f'results/{model}_{job}' if not (model == 'gemma' and job == 'dpo') else 'results/gemma_dpo'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=MODELS, required=True); ap.add_argument('--job', choices=JOBS, required=True)
    a = ap.parse_args()
    os.environ['GS_MODEL'] = a.model
    BASE = HOME / 'idea3_gemma_speech'
    q.MODEL_ID, q.REV, min_gb = MODELS[a.model]
    q.SCRIPT, stages, hours = JOBS[a.job]
    q.SUB = sub_for(a.model, a.job)
    q.DRIVE = BASE / f'state_{a.model}_{a.job}'; q.STATE = q.DRIVE / 'results'
    q.ROOT = BASE / 'gs'; q.CODE = q.ROOT / 'final_omni'; q.VENV = BASE / 'gsenv'; q.PY = q.VENV / 'bin/python'
    q.HF = BASE / 'hf'; q.P777 = q.ROOT / 'p777'
    bundle = HERE / q.BUNDLE
    q.DRIVE.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    mark = q.ROOT / '.bundle_sha'
    if q.ROOT.exists() and (not mark.exists() or mark.read_text() != sha):
        q.ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', 'xzf', str(bundle), '-C', str(q.ROOT)], check=True)
        (q.ROOT / '.unpacked').touch()
        print('unpacked this bundle over an earlier one (environment, models and results kept)', flush=True)
    shutil.copy(bundle, q.DRIVE / q.BUNDLE)
    if shutil.which('uv') is None:
        subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)
        os.environ['PATH'] = str(HOME / '.local/bin') + ':' + os.environ['PATH']
    _sh = q.sh
    q.sh = lambda cmd, **k: None if cmd == 'pip -q install uv' else _sh(cmd, **k)
    q.gpu_check(min_gb=min_gb)
    q.setup()
    mark.write_text(sha)
    if a.job in ('dpo', 'fixeval') and subprocess.run([str(q.PY), '-c', 'import peft'], capture_output=True).returncode:
        q.sh(f'uv pip install -q --python {q.PY} {PEFT}')
    print(f'=== {a.model} {a.job}: {q.SCRIPT} {stages}, results in {q.SUB}', flush=True)
    q.run(stages, hours=hours)
    q.sync()
    G = q.CODE / q.SUB
    z = BASE / f'{a.model}_{a.job}_results_{time.strftime("%m%d_%H%M")}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(G.rglob('*')):
            if p.is_file() and p.name not in ('last.pt', 'ref.json') and 'acts' not in p.parts:
                f.write(p, Path(G.name) / p.relative_to(G))
    print('RESULTS ZIP:', z, round(z.stat().st_size / 1e6, 1), 'MB')
    for rep in ('report.md', 'report_val.md'):
        if (G / rep).exists():
            print((G / rep).read_text())


if __name__ == '__main__':
    main()
