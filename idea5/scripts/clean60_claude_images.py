"""Clean-60 image conditions for Claude: one fresh, tool-free `claude -p` call per (item, condition).

Images are sent inline as message content (stream-json input), so no Read tool, no files, no web,
and no context shared between questions.
"""
import base64
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean60_run import CONTEXT, IMAGES, OUT, options

MODELS = {'opus': 'claude-opus-5-5', 'sonnet': 'claude-sonnet-5-5'}
WORKDIR = tempfile.mkdtemp(prefix='clean60-empty-')


def call(job):
    name, cond, item = job
    target = OUT / 'results/claude_single' / f"{name}__{cond}__{item['id']}.json"
    if target.exists():
        return
    prompt = f"{CONTEXT[cond]}\n\n{options(item)}\n\nReply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': base64.b64encode(Path(item['images'][k]).read_bytes()).decode()}} for k in IMAGES[cond]]
    message = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}}) + '\n'
    cmd = ['claude', '-p', '--model', MODELS[name], '--tools', '', '--strict-mcp-config', '--no-session-persistence',
           '--system-prompt', 'You answer multiple-choice questions.', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose']
    out = ''
    for _ in range(3):
        out = subprocess.run(cmd, input=message, text=True, capture_output=True, timeout=300, cwd=WORKDIR).stdout
        init, meta = None, None
        for line in out.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') == 'system' and d.get('subtype') == 'init':
                init = d
            if d.get('type') == 'result':
                meta = d
        if not meta or not init:
            continue
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', meta.get('result') or '', re.S)
        if m:
            usage = meta['usage']
            target.write_text(json.dumps({'id': item['id'], 'condition': cond, 'model': init['model'], 'action_position': int(m.group(1)),
                                          'justification_position': int(m.group(2)), 'raw': meta['result'], 'num_turns': meta['num_turns'],
                                          'tools_available': init['tools'] if isinstance(init['tools'], int) else len(init['tools']),
                                          'mcp_servers': init['mcp_servers'] if isinstance(init['mcp_servers'], int) else len(init['mcp_servers']),
                                          'web_search_requests': usage['server_tool_use']['web_search_requests'],
                                          'web_fetch_requests': usage['server_tool_use']['web_fetch_requests'],
                                          'images_sent': len(content),
                                          'context_tokens': usage['input_tokens'] + usage['cache_read_input_tokens'] + usage['cache_creation_input_tokens']}))
            return
    target.with_suffix('.failed').write_text(out[-4000:])


if __name__ == '__main__':
    (OUT / 'results/claude_single').mkdir(parents=True, exist_ok=True)
    tasks = json.loads((OUT / 'tasks.json').read_text())
    jobs = [(m, c, t) for m in MODELS for c in ['rgb', 'geo'] for t in tasks]
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(call, jobs))
    done = list((OUT / 'results/claude_single').glob('*__rgb__*.json')) + list((OUT / 'results/claude_single').glob('*__geo__*.json'))
    print(len(done), 'of', len(jobs), 'image calls answered;', len(list((OUT / 'results/claude_single').glob('*.failed'))), 'failed')
