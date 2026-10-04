"""Clean-60 text-only check: one fresh, tool-free `claude -p` call per item (no files, no web, no shared context)."""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean60_run import CONTEXT, OUT, options

CONTEXT['textforced'] = CONTEXT['text'] + " Even though the situation is not shown, pick the options most likely to be correct; do not choose a 'None of these' option merely because the situation is not shown."
MODELS = {'opus': 'claude-opus-5-5', 'sonnet': 'claude-sonnet-5-5'}
WORKDIR = tempfile.mkdtemp(prefix='clean60-empty-')


def call(job):
    name, cond, item = job
    target = OUT / 'results/claude_single' / f"{name}__{cond}__{item['id']}.json"
    if target.exists():
        return
    prompt = f"{CONTEXT[cond]}\n\n{options(item)}\n\nReply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
    cmd = ['claude', '-p', '--model', MODELS[name], '--tools', '', '--strict-mcp-config', '--no-session-persistence',
           '--system-prompt', 'You answer multiple-choice questions.', '--output-format', 'json']
    for _ in range(3):
        out = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=300, cwd=WORKDIR).stdout
        try:
            meta = json.loads(out)
        except json.JSONDecodeError:
            continue
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', meta.get('result', ''), re.S)
        if m:
            target.write_text(json.dumps({'id': item['id'], 'condition': cond, 'model': list(meta['modelUsage']), 'action_position': int(m.group(1)),
                                          'justification_position': int(m.group(2)), 'raw': meta['result'], 'num_turns': meta['num_turns'],
                                          'web_search_requests': meta['usage']['server_tool_use']['web_search_requests'],
                                          'web_fetch_requests': meta['usage']['server_tool_use']['web_fetch_requests'],
                                          'input_tokens': meta['usage']['input_tokens'], 'cache_read_input_tokens': meta['usage']['cache_read_input_tokens']}))
            return
    target.with_suffix('.failed').write_text(out)


if __name__ == '__main__':
    (OUT / 'results/claude_single').mkdir(parents=True, exist_ok=True)
    tasks = json.loads((OUT / 'tasks.json').read_text())
    jobs = [(m, c, t) for m in MODELS for c in ['text', 'textforced'] for t in tasks]
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(call, jobs))
