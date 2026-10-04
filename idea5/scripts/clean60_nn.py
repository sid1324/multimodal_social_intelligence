"""Clean-60 pilot, forced-choice variant: every prompt explicitly forbids the "None of these" options.

Conditions: text_nn (options only), rgb_nn (+ RGB frames), geo_nn (+ RGB frames, depth, segmentation).
One isolated call per (model, condition, item). GPT calls keep the Codex CLI event log so that
tool use can be checked for every scored call, not just a sample.

Usage: clean60_nn.py gpt <model> [<model> ...]     (codex exec)
       clean60_nn.py claude                        (claude -p, tools disabled; Opus and Sonnet)
"""
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean60_run as c
import clean60_claude_images as ci

NO_NONE = " Do not answer 'None of these': always choose one of the other four actions and one of the other four justifications."
BASE = {'text_nn': 'text', 'rgb_nn': 'rgb', 'geo_nn': 'geo'}
for cond, base in BASE.items():
    c.CONTEXT[cond] = c.CONTEXT[base] + NO_NONE
    c.IMAGES[cond] = c.IMAGES[base]
FOLDER = {'gpt-5.6-sol': 'gpt', 'gpt-5.5': 'gpt-5.5', 'gpt-6-sol': 'gpt-6-sol'}


def gpt_call(job):
    model, cond, item, workdir = job
    folder = c.OUT / 'results' / FOLDER[model]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{cond}__{item['id']}.json"
    if target.exists():
        return
    prompt = f"{c.CONTEXT[cond]}\n\n{c.options(item)}\n\nDo not use any tools or read any files. Reply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
    cmd = ['codex', 'exec', '-m', model, '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral', '-C', workdir, '--json']
    for k in c.IMAGES[cond]:
        cmd += ['-i', item['images'][k]]
    out = ''
    for _ in range(3):
        out = subprocess.run(cmd + ['-'], input=prompt, text=True, capture_output=True, timeout=600).stdout
        events, reply = Counter(), ''
        for line in out.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get('type', '?')
            if isinstance(e.get('item'), dict):
                kind += ':' + e['item'].get('type', '?')
                if e['item'].get('type') == 'agent_message':
                    reply = e['item'].get('text', '')
            events[kind] += 1
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', reply, re.S)
        if m:
            target.write_text(json.dumps({'id': item['id'], 'condition': cond, 'model': model, 'action_position': int(m.group(1)),
                                          'justification_position': int(m.group(2)), 'raw': reply, 'events': dict(events)}))
            return
    target.with_suffix('.failed').write_text(out[-4000:])


def main():
    tasks = json.loads((c.OUT / 'tasks.json').read_text())
    if sys.argv[1] == 'gpt':
        workdir = tempfile.mkdtemp(prefix='clean60-empty-')
        jobs = [(m, cond, t, workdir) for m in sys.argv[2:] for cond in BASE for t in tasks]
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(gpt_call, jobs))
        for m in sys.argv[2:]:
            done = list((c.OUT / 'results' / FOLDER[m]).glob('*_nn__*.json'))
            print(m, len(done), 'of', 3 * len(tasks), 'answered;', len(list((c.OUT / 'results' / FOLDER[m]).glob('*_nn__*.failed'))), 'failed')
    else:
        jobs = [(name, cond, t) for name in ci.MODELS for cond in BASE for t in tasks]
        with ThreadPoolExecutor(6) as pool:
            list(pool.map(ci.call, jobs))
        done = list((c.OUT / 'results/claude_single').glob('*_nn__*.json'))
        print('claude', len(done), 'of', len(jobs), 'answered;', len(list((c.OUT / 'results/claude_single').glob('*_nn__*.failed'))), 'failed')


if __name__ == '__main__':
    main()
