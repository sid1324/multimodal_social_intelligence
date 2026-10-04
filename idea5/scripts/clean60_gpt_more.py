"""Clean-60 pilot for additional GPT models through `codex exec`, one stateless call per (item, condition)."""
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean60_run as c

c.CONTEXT['textforced'] = c.CONTEXT['text'] + " Even though the situation is not shown, pick the options most likely to be correct; do not choose a 'None of these' option merely because the situation is not shown."
c.IMAGES['textforced'] = []
tasks = json.loads((c.OUT / 'tasks.json').read_text())
workdir = tempfile.mkdtemp(prefix='clean60-empty-')
jobs = [(m, cond, t, workdir) for m in sys.argv[1:] for cond in ['text', 'textforced', 'rgb', 'geo'] for t in tasks]
with ThreadPoolExecutor(8) as pool:
    list(pool.map(c.gpt_call, jobs))
for m in sys.argv[1:]:
    print(m, len(list((c.OUT / 'results' / m).glob('*.json'))), 'answered,', len(list((c.OUT / 'results' / m).glob('*.failed'))), 'failed')
