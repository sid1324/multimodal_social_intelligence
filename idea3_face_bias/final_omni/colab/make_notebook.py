"""Writes colab/idea3_final.ipynb (Colab notebook)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}
cells = [
md("""# Idea 3 final: run on Colab

**Before you start**
1. Runtime > Change runtime type > **A100 GPU**, with **High RAM** on (this gives the 80 GB card). Save.
2. Put `idea3_bundle.tar.gz` in **My Drive/idea3_final/**.
3. Run the cells from top to bottom. If Colab disconnects, run them all again: it continues where it stopped.

**Cost:** roughly 7 to 9 compute units per hour on A100 80 GB (measured by users, not published by Google). The run stops by itself after `MAX_HOURS` (default 11, about 75 to 100 units) and releases the GPU.

**What it does, in order:** machine check against our Delta results, blank-image and wrong-image check, steering sweep (validation), DPO training rows, DPO safety check, DPO pairs, DPO-text, DPO-multi, DPO validation. It writes a *proposed* steering freeze for review.

**The test split stays locked.** Nothing here can run it.
"""),
code("""from google.colab import drive
drive.mount('/content/drive')
import os, sys
assert os.path.exists('/content/drive/MyDrive/idea3_final/idea3_bundle.tar.gz'), 'Put idea3_bundle.tar.gz in My Drive/idea3_final'
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
