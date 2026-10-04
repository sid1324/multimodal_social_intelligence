"""Clean-60 pilot on the local Qwen3-VL-32B (8-bit MLX): text / textforced / rgb / geo, one stateless call each."""
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

sys.path.insert(0, str(Path(__file__).parent))
from clean60_run import CONTEXT, IMAGES, OUT, options

MODEL = str(OUT.parents[1] / 'models/Qwen3-VL-32B-Instruct-8bit')
CONTEXT['textforced'] = CONTEXT['text'] + " Even though the situation is not shown, pick the options most likely to be correct; do not choose a 'None of these' option merely because the situation is not shown."
IMAGES['textforced'] = []


def main(limit):
    tasks = json.loads((OUT / 'tasks.json').read_text())[:limit]
    target = OUT / 'results/qwen32b.json'
    results = json.loads(target.read_text()) if target.exists() else []
    done = {(r['id'], r['condition']) for r in results}
    model, processor = load(MODEL)
    config = load_config(MODEL)
    for item in tasks:
        for cond in ['text', 'textforced', 'rgb', 'geo']:
            if (item['id'], cond) in done:
                continue
            images = [item['images'][k] for k in IMAGES[cond]]
            prompt = f"{CONTEXT[cond]}\n\n{options(item)}\n\nReply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
            mx.random.seed(42)
            formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
            started = time.time()
            output = generate(model, processor, formatted, image=images or None, max_tokens=32, temperature=0., verbose=False)
            reply = output.text if hasattr(output, 'text') else str(output)
            m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', reply, re.S)
            results.append({'id': item['id'], 'condition': cond, 'action_position': int(m.group(1)) if m else None,
                            'justification_position': int(m.group(2)) if m else None, 'raw': reply, 'seconds': time.time() - started})
            target.write_text(json.dumps(results, indent=1))
            print(len(results), cond, reply.strip()[:60].replace('\n', ' '), f'{time.time() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
