import json
from pathlib import Path
H = Path(__file__).resolve().parent
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}
cells = [
md("""# Qwen3-VL 8B: does text steering transfer to captions in the image?

**Start:** Select Kernel > Colab > New Colab Server > GPU > **A100**, or **L4** if no A100 (24 GB fits; slower, cheaper per hour). Right-click `qwen_transfer_bundle.tar.gz` in the explorer > **Upload to Colab**. Command Palette > **Colab: Mount Google Drive to Server**. Then run the cells top to bottom. File > Auto Save on, so Claude can read the output.

**Cost:** about 5.5 to 7 units per hour. Estimated 2 to 4 hours in total. If Colab disconnects, run the cells again; it continues. **When done: Colab: Remove Server.**

**On the Colab website instead:** File > Upload notebook (this file). Runtime > Change runtime type > A100 (or L4). Put `qwen_transfer_bundle.tar.gz` in Google Drive at `My Drive/idea3_qwen_transfer/`. The first cell asks to connect Drive. At the end, the zip cell makes one file to send back.

Nothing here can run the test scenes."""),
code("""import os, subprocess
print(subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /content | tail -1; python3 --version; ls /content', shell=True, capture_output=True, text=True).stdout)
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive; drive.mount('/content/drive')   # Colab website; in VS Code use the Command Palette instead
    except Exception as ex:
        print('Drive not mounted here:', ex)
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
Approved by Yash, Nov 25 22:10 CT: the success rule in `QWEN_TRANSFER_PLAN.md` and the caption line. Runs only if Part 1 passed every check and the bias gate. About 1 to 2 hours on an A100."""),
code("""RULE_APPROVED = True       # Yash approved the success rule and caption line, Nov 25 22:10 CT
if RULE_APPROVED:
    from pathlib import Path
    Path('/content/qt/qwen_transfer/results/PREREG_OK').write_text('approved by Yash, Nov 25 22:10 CT\\n')
    q.run(['exp', 'report val:tr'], hours=6)
else:
    print('Not started: set RULE_APPROVED = True after reading the rule.')"""),
code("""# optional: see the tables any time
q.stage('gate', None); q.stage('report val:tr', None)"""),
code("""# send-back file: zips all results (no images, no model). Saved to Drive (My Drive/idea3_qwen_transfer/) and /content
import shutil, time
z = shutil.make_archive(f'/content/qt_results_{time.strftime("%m%d_%H%M")}', 'zip', '/content/qt/qwen_transfer/results')
if os.path.exists('/content/drive/MyDrive'): shutil.copy(z, '/content/drive/MyDrive/idea3_qwen_transfer/')
print('send this file:', z, round(os.path.getsize(z) / 1e6, 1), 'MB')"""),
code("""# when finished: release the GPU so it stops using units
q.sync(); q.release()"""),
]
(H / 'qwen_transfer.ipynb').write_text(json.dumps({'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}, indent=1))
