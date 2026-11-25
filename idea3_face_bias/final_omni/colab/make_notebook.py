"""Writes colab/idea3_final.ipynb (Colab notebook)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}
cells = [
md("""# Idea 3 final on Colab (VS Code extension)

1. Select Kernel > Colab > New Colab Server > GPU > **A100**, shape **High RAM** (80 GB card).
2. In the explorer, right-click `idea3_bundle.tar.gz` > **Upload to Colab**.
3. Command Palette > **Colab: Mount Google Drive to Server** (so progress survives a disconnect).
4. Run the cells top to bottom. If it disconnects, run them again: it continues where it stopped.
5. Keep this VS Code window open and **save the notebook now and then** (File > Auto Save is better): Claude reads the saved output.
6. **When finished, Command Palette > Colab: Remove Server.** It costs units until you do.

Order: match check against Delta, steering sweep (validation), DPO rows, DPO safety check, pairs, DPO-text, DPO-multi, DPO validation. Stops after `MAX_HOURS`. **The test split stays locked.**
"""),
code("""import os, subprocess, sys, shutil
print(subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /content | tail -1; python3 --version; ls /content', shell=True, capture_output=True, text=True).stdout)
print('Drive mounted:', os.path.exists('/content/drive/MyDrive'))
"""),
code('%%writefile /content/i3_colab.py\n' + (H / 'i3_colab.py').read_text()),
code("""# 1. Spending limit and behaviour
MAX_HOURS = 11        # stop after this many hours (resumable, so you can continue later)
AUTO_STOP = True      # release the GPU at the end so it stops using compute units
sys.path.insert(0, '/content')
import i3_colab as i3
i3.gpu_check()        # stops here if the GPU is too small
"""),
code("""# 2. Setup: unpack, restore earlier progress, install exact package versions, download the model (about 10 to 25 min the first time)
i3.setup()
print(i3.check())
"""),
code("""# 3. Run everything that is ready (leave this tab open, or use background execution if your plan has it)
i3.run(max_hours=MAX_HOURS, auto_stop=AUTO_STOP)
"""),
code("""# Optional: look at progress at any time while cell 3 is NOT running
i3.status(); print(i3.check())
"""),
]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': []},
      'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
(H / 'idea3_final.ipynb').write_text(json.dumps(nb, indent=1))
