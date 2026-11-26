import json
from pathlib import Path
H = Path(__file__).resolve().parent
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}
cells = [
md("""# Qwen3-VL 8B: does text steering transfer to captions in the image?

**Start:** Select Kernel > Colab > New Colab Server > GPU > **A100**, or **L4** if no A100 (24 GB fits; slower, cheaper per hour). Right-click `qwen_transfer_bundle.tar.gz` in the explorer > **Upload to Colab**. Command Palette > **Colab: Mount Google Drive to Server**. Then run the cells top to bottom. File > Auto Save on, so Claude can read the output.

**Cost:** about 5.5 to 7 units per hour. Estimated 2 to 4 hours in total. If Colab disconnects, run the cells again; it continues. **When done: Colab: Remove Server.**

Nothing here can run the test scenes."""),
code("""import os, subprocess
print(subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /content | tail -1; python3 --version; ls /content', shell=True, capture_output=True, text=True).stdout)
print('Drive mounted:', os.path.exists('/content/drive/MyDrive'))"""),
code('%%writefile /content/qt_colab.py\n' + (H / 'qt_colab.py').read_text()),
code("""import sys; sys.path.insert(0, '/content')
import qt_colab as q
q.gpu_check()   # stops here if the GPU is wrong
q.setup()       # about 10 to 20 minutes the first time"""),
md("""## Part 1: checks and base bias (no steering). About 1 to 1.5 hours
1. reproduce midterm text answers (must match 98%)  2. can it read the caption (90%)  3. base answers on image captions  4. bias gate"""),
code("""q.run(['repro', 'perc', 'base', 'gate'], hours=4)"""),
md("""## Part 2: the steered runs
Read the success rule in `QWEN_TRANSFER_PLAN.md`. **Only if you agree, set the flag below to True.**"""),
code("""RULE_APPROVED = False      # set to True only after Yash has read and approved the success rule
if RULE_APPROVED:
    from pathlib import Path
    Path('/content/qt/qwen_transfer/results/PREREG_OK').write_text('approved by Yash\\n')
    q.run(['exp', 'report val:tr'], hours=6)
else:
    print('Not started: set RULE_APPROVED = True after reading the rule.')"""),
code("""# optional: see the tables any time
q.stage('gate', None); q.stage('report val:tr', None)"""),
code("""# when finished: release the GPU so it stops using units
q.sync(); q.release()"""),
]
(H / 'qwen_transfer.ipynb').write_text(json.dumps({'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}, indent=1))
