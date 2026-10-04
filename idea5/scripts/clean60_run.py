"""Build gold-free task files for the clean-60 pilot and run the GPT arm through `codex exec`.

Usage: clean60_run.py tasks            write Claude batch files
       clean60_run.py gpt [model]      one stateless codex exec call per (item, condition)
"""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
TASK = ('Choose the single most socially appropriate next action for the camera wearer from the action options, '
        'then choose the justification that best fits the action you chose.')
CONTEXT = {
    'text': 'A person wearing a head-mounted camera is in an everyday situation. You are NOT shown the situation; no image or description is available. Answer from the options alone. ' + TASK,
    'rgb': 'The image is a left-to-right grid of five chronological frames from an egocentric (head-mounted) camera, captured before the camera wearer\'s next action. ' + TASK,
    'geo': 'Three images are provided. Image 1: a left-to-right grid of five chronological RGB frames from an egocentric (head-mounted) camera, captured before the camera wearer\'s next action. '
           'Image 2: estimated relative depth for the same five frames (brighter is nearer; not metric; can be inaccurate). '
           'Image 3: instance segmentation for the same five frames (coloured masks with predicted COCO class labels; black is unsegmented, not empty space; labels can be wrong). ' + TASK,
}
IMAGES = {'text': [], 'rgb': ['rgb'], 'geo': ['rgb', 'depth', 'segmentation']}


def options(item):
    lines = ['Action options:'] + [f'{i}. {o}' for i, o in enumerate(item['action_options'], 1)]
    lines += ['', 'Justification options:'] + [f'{i}. {o}' for i, o in enumerate(item['justification_options'], 1)]
    return '\n'.join(lines)


def write_tasks():
    tasks = json.loads((OUT / 'tasks.json').read_text())
    (OUT / 'claude_tasks').mkdir(exist_ok=True)
    (OUT / 'results').mkdir(exist_ok=True)
    for cond in CONTEXT:
        batches = [tasks] if cond == 'text' else [tasks[i:i + 20] for i in range(0, 60, 20)]
        for b, batch in enumerate(batches, 1):
            items = [{'case_number': t['case_number'], 'id': t['id'], 'images_in_order': [t['images'][k] for k in IMAGES[cond]],
                      'action_options': t['action_options'], 'justification_options': t['justification_options']} for t in batch]
            (OUT / 'claude_tasks' / f'{cond}_b{b}.json').write_text(json.dumps({'condition': cond, 'context': CONTEXT[cond], 'indexing': '1-based option positions', 'items': items}, indent=1))


def gpt_call(job):
    model, cond, item, workdir = job
    folder = OUT / 'results' / ('gpt' if model == 'gpt-5.6-sol' else model)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{cond}__{item['id']}.json"
    if target.exists():
        return
    prompt = f"{CONTEXT[cond]}\n\n{options(item)}\n\nDo not use any tools or read any files. Reply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
    cmd = ['codex', 'exec', '-m', model, '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral', '-C', workdir]
    for k in IMAGES[cond]:
        cmd += ['-i', item['images'][k]]
    for _ in range(3):
        with tempfile.NamedTemporaryFile('r', suffix='.txt') as last:
            subprocess.run(cmd + ['-o', last.name, '-'], input=prompt, text=True, capture_output=True, timeout=600)
            reply = Path(last.name).read_text()
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', reply, re.S)
        if m:
            target.write_text(json.dumps({'id': item['id'], 'condition': cond, 'model': model, 'action_position': int(m.group(1)), 'justification_position': int(m.group(2)), 'raw': reply}))
            return
    target.with_suffix('.failed').write_text(reply)


def run_gpt(model):
    tasks = json.loads((OUT / 'tasks.json').read_text())
    (OUT / 'results' / 'gpt').mkdir(parents=True, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix='clean60-empty-')
    jobs = [(model, cond, item, workdir) for cond in CONTEXT for item in tasks]
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(gpt_call, jobs))
    print(len(list((OUT / 'results' / 'gpt').glob('*.json'))), 'of', len(jobs), 'answered')


if __name__ == '__main__':
    write_tasks() if sys.argv[1] == 'tasks' else run_gpt(sys.argv[2] if len(sys.argv) > 2 else 'gpt-5.6-sol')
