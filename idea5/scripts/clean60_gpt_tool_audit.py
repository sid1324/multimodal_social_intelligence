"""Audit whether GPT calls made through `codex exec` use any tool: re-run a sample with --json and list event/item types."""
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean60_run as c

c.CONTEXT['textforced'] = c.CONTEXT['text'] + " Even though the situation is not shown, pick the options most likely to be correct; do not choose a 'None of these' option merely because the situation is not shown."
c.IMAGES['textforced'] = []
tasks = json.loads((c.OUT / 'tasks.json').read_text())
rng = random.Random(7)
jobs = [(m, cond, t) for m in ['gpt-5.5', 'gpt-5.6-sol', 'gpt-6-sol'] for cond in ['textforced', 'rgb', 'geo'] for t in rng.sample(tasks, 5)]
workdir = tempfile.mkdtemp(prefix='clean60-empty-')


def audit(job):
    model, cond, item = job
    prompt = f"{c.CONTEXT[cond]}\n\n{c.options(item)}\n\nDo not use any tools or read any files. Reply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
    cmd = ['codex', 'exec', '-m', model, '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral', '-C', workdir, '--json']
    for k in c.IMAGES[cond]:
        cmd += ['-i', item['images'][k]]
    out = subprocess.run(cmd + ['-'], input=prompt, text=True, capture_output=True, timeout=600).stdout
    kinds = Counter()
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        kinds[e.get('type', '?') + ((':' + e['item'].get('type', '?')) if isinstance(e.get('item'), dict) else '')] += 1
    return model, cond, dict(kinds)


with ThreadPoolExecutor(8) as pool:
    rows = list(pool.map(audit, jobs))
total = Counter()
for model, cond, kinds in rows:
    total.update(kinds)
print('calls audited:', len(rows))
print('event/item types across all audited calls:', dict(total))
tool_like = {k: v for k, v in total.items() if any(w in k for w in ['command', 'exec', 'tool', 'search', 'mcp', 'file', 'patch', 'web'])}
print('tool-like events:', tool_like or 'none')
(c.OUT / 'gpt_tool_audit.json').write_text(json.dumps({'calls': [{'model': m, 'condition': cond, 'events': k} for m, cond, k in rows], 'totals': dict(total), 'tool_like': tool_like}, indent=1))
