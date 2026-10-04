"""Forced-choice variant of the clean-60 pilot on the local Qwen3-VL-32B (8-bit MLX), one stateless call each."""
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
import clean60_run as c

NO_NONE = " Do not answer 'None of these': always choose one of the other four actions and one of the other four justifications."
MODEL = str(c.OUT.parents[1] / 'models/Qwen3-VL-32B-Instruct-8bit')
CONDS = {'text_nn': 'text', 'rgb_nn': 'rgb', 'geo_nn': 'geo'}


def main():
    tasks = json.loads((c.OUT / 'tasks.json').read_text())
    target = c.OUT / 'results/qwen32b_nn.json'
    results = json.loads(target.read_text()) if target.exists() else []
    done = {(r['id'], r['condition']) for r in results}
    model, processor = load(MODEL)
    config = load_config(MODEL)
    for item in tasks:
        for cond, base in CONDS.items():
            if (item['id'], cond) in done:
                continue
            images = [item['images'][k] for k in c.IMAGES[base]]
            prompt = f"{c.CONTEXT[base] + NO_NONE}\n\n{c.options(item)}\n\nReply with only this JSON and nothing else: {{\"action\": <1-5>, \"justification\": <1-5>}}"
            mx.random.seed(42)
            formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
            started = time.time()
            output = generate(model, processor, formatted, image=images or None, max_tokens=32, temperature=0., verbose=False)
            reply = output.text if hasattr(output, 'text') else str(output)
            m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', reply, re.S)
            results.append({'id': item['id'], 'condition': cond, 'action_position': int(m.group(1)) if m else None,
                            'justification_position': int(m.group(2)) if m else None, 'raw': reply, 'seconds': time.time() - started})
            target.write_text(json.dumps(results, indent=1))
    print('qwen rows', len(results))


if __name__ == '__main__':
    main()
