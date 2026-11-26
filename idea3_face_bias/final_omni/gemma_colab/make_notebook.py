"""Writes gemma_colab/gemma_speech.ipynb (Colab notebook for the Gemma 4 speech-transfer test)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}
cells = [
md("""# Gemma 4 12B: does steering built from typed remarks also work on spoken remarks?

**On the Colab website:** put `gemma_speech_bundle.tar.gz` in Google Drive at `My Drive/idea3_gemma_speech/`. File > Upload notebook (this file). Runtime > Change runtime type > **A100** (or H100 / G4). Then **Runtime > Run all**. Allow Drive access when asked.

**In VS Code (Colab extension):** Select Kernel > Colab > New Colab Server > GPU > A100. Right-click the bundle > Upload to Colab. Command Palette > Colab: Mount Google Drive to Server. Run all cells.

**What it does:** checks the model, checks it can hear the spoken remark, measures bias in typed and spoken remarks, builds the appearance direction from typed remarks only, picks the steering setting on typed remarks, then applies it to spoken remarks, compared with a direction built from speech (the ceiling) and with control directions (weather mood, shuffled labels, random) and a written rule. An exploratory face arm shows a second image with a face rated more or less trustworthy. Levels: L1 (main) and L2.

**Cost:** roughly 6 to 9 hours on an A100 (about 40 to 60 units). If Colab disconnects, run all cells again: it continues where it stopped. It stops by itself if a check fails. **The last cells make one zip to send back, then release the GPU.**

Nothing here can run the test scenes."""),
code("""import os, subprocess
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive; drive.mount('/content/drive')   # Colab website; in VS Code use the Command Palette instead
    except Exception as ex:
        print('Drive not mounted here:', ex)
print(subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /content | tail -1; python3 --version', shell=True, capture_output=True, text=True).stdout)
print('Drive mounted:', os.path.exists('/content/drive/MyDrive'))"""),
code('%%writefile /content/gs_colab.py\n' + (H / 'gs_colab.py').read_text()),
code("""import sys; sys.path.insert(0, '/content')
import gs_colab as q
q.gpu_check()   # stops here if the GPU is too small
q.setup()       # about 15 to 25 minutes the first time (packages, images, 24 GB model)"""),
md("""## Part A: checks and bias in typed and spoken remarks (no steering). About 1.5 to 2.5 hours
1. model check (letters, greedy answer, steering hook)  2. can it hear the remark (90%)  3. unsteered answers on 300 training scenes  4. bias gate per level"""),
code("""q.run(['check', 'perc', 'g2', 'gate'], hours=5)"""),
md("""## Part B: steering built from typed remarks, applied to spoken remarks, plus the face arm. About 4 to 6 hours
Runs only if Part A passed. Levels that failed the gate are skipped automatically."""),
code("""G = '/content/gs/final_omni/results/gemma/'
if os.path.exists(G + 'STOP.md'):
    print('Stopped in Part A:', open(G + 'STOP.md').read())
elif not os.path.exists(G + 'gate.json'):
    print('Part A is not finished (time limit or disconnect): run all cells again; it continues where it stopped.')
else:
    q.run(['directions', 'sweep', 'select', 'transfer', 'faces', 'report'], hours=9)"""),
code("""# send-back file: answer rows, directions, summaries (no model, no images, no per-scene activations). Saved to Drive and /content
import shutil, time
q.sync()
import zipfile
from pathlib import Path
src = Path('/content/gs/final_omni/results/gemma'); z = f'/content/gemma_results_{time.strftime("%m%d_%H%M")}.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
    for p in sorted(src.rglob('*')):
        if p.is_file() and 'acts' not in p.parts:   # per-scene activation files stay on Drive (large)
            f.write(p, p.relative_to(src))
if os.path.exists('/content/drive/MyDrive'): shutil.copy(z, '/content/drive/MyDrive/idea3_gemma_speech/')
print('send this file:', z, round(os.path.getsize(z) / 1e6, 1), 'MB')"""),
code("""# release the GPU so it stops using units
q.release()"""),
]
(H / 'gemma_speech.ipynb').write_text(json.dumps({'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}, indent=1))
