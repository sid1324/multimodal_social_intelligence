"""Depth-only condition for the clean-60 forced-choice run: answer options plus the depth grid, no RGB frames.

Condition dep_nn. One isolated call per item, same instructions as the other forced-choice conditions.
Usage: clean60_depth_only.py gpt <model> [...]  |  clean60_depth_only.py claude  |  clean60_depth_only.py qwen
"""
import json
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean60_nn as n

c = n.c
DEPTH_ONLY = ("The image is a left-to-right grid of five chronological frames from an egocentric (head-mounted) camera, captured before the camera wearer's next action. "
              "The frames are shown only as estimated relative depth maps (brighter is nearer; not metric; can be inaccurate); no RGB image is provided. " + c.TASK)
c.CONTEXT['dep_nn'] = DEPTH_ONLY + n.NO_NONE
c.IMAGES['dep_nn'] = ['depth']


def run_qwen(tasks):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model_path = str(c.OUT.parents[1] / 'models/Qwen3-VL-32B-Instruct-8bit')
    target = c.OUT / 'results/qwen32b_depth_only.json'
    results = json.loads(target.read_text()) if target.exists() else []
    done = {r['id'] for r in results}
    model, processor = load(model_path)
    config = load_config(model_path)
    for item in tasks:
        if item['id'] in done:
            continue
        images = [item['images']['depth']]
        prompt = f"{c.CONTEXT['dep_nn']}\n\n{c.options(item)}\n\nReply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
        mx.random.seed(42)
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        started = time.time()
        output = generate(model, processor, formatted, image=images, max_tokens=32, temperature=0., verbose=False)
        reply = output.text if hasattr(output, 'text') else str(output)
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', reply, re.S)
        results.append({'id': item['id'], 'condition': 'dep_nn', 'action_position': int(m.group(1)) if m else None,
                        'justification_position': int(m.group(2)) if m else None, 'raw': reply, 'seconds': time.time() - started})
        target.write_text(json.dumps(results, indent=1))
    print('qwen depth-only rows', len(results))


def main():
    tasks = json.loads((c.OUT / 'tasks.json').read_text())
    if sys.argv[1] == 'gpt':
        workdir = tempfile.mkdtemp(prefix='clean60-empty-')
        jobs = [(m, 'dep_nn', t, workdir) for m in sys.argv[2:] for t in tasks]
        with ThreadPoolExecutor(12) as pool:
            list(pool.map(n.gpt_call, jobs))
        for m in sys.argv[2:]:
            folder = c.OUT / 'results' / n.FOLDER[m]
            print(m, len(list(folder.glob('dep_nn__*.json'))), 'of', len(tasks), 'answered;', len(list(folder.glob('dep_nn__*.failed'))), 'failed')
    elif sys.argv[1] == 'claude':
        jobs = [(name, 'dep_nn', t) for name in n.ci.MODELS for t in tasks]
        with ThreadPoolExecutor(6) as pool:
            list(pool.map(n.ci.call, jobs))
        folder = c.OUT / 'results/claude_single'
        print('claude', len(list(folder.glob('*__dep_nn__*.json'))), 'of', len(jobs), 'answered;', len(list(folder.glob('*__dep_nn__*.failed'))), 'failed')
    else:
        run_qwen(tasks)


if __name__ == '__main__':
    main()
