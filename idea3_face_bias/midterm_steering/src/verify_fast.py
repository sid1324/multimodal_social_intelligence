"""Check that batched shared-prefix scoring (--fast) matches one-prompt-at-a-time scoring.
v2 (2026-10-25): v1 failed only on max |delta logp| (0.75 = three bf16 logit steps of 0.25). Logits are now
computed in float32 from the final hidden state for both paths, and the pass criterion compares the fast path
with a pure rounding-noise reference (the same prompt as a duplicated batch of two); see Qwen.verify_vs_noise.
The strict v1 numbers are still reported under 'strict_v1'.
Train items only. Cases: base model, constitution system prompt, and an active ablation hook
(seeded random unit direction on layers 14-35, alpha 1). Writes results/fast_check.json; 'ok' requires the
same argmax on every prompt and max |delta letter log-prob| <= 0.05 (bf16 rounding differs between
batch shapes, so exact equality is not expected). Also reports the speed of both paths.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, load_annotations, load_item_table, load_splits, load_cues, constitution, permuted_options, prompt_text, image_path, gate_items
from model import Qwen
from run_gate import GATE_CONDS


def main(n_items=6):
    ann, table, splits, cues = load_annotations(), load_item_table(), load_splits(), load_cues()
    items = gate_items(splits, table, 300)[-n_items:]
    q = Qwen()
    cases = lambda system: [([Image.open(image_path(k)).convert('RGB')], [prompt_text(ann[k], cues[c]) for c in GATE_CONDS],
                             len(permuted_options(ann[k])), system) for k in items]
    res = {'strict_v1': {}}
    rnd = np.random.default_rng(0).standard_normal((q.n_layers, q.model.config.text_config.hidden_size))
    for name, system, abl in (('base', None, False), ('constitution', constitution(), False), ('ablation_random_dir', None, True)):
        q.set_ablation(rnd, 1.0, range(14, 36)) if abl else q.set_ablation(None)
        res[name] = q.verify_vs_noise(cases(system))
        res['strict_v1'][name] = q.verify_fast(cases(system))
    q.set_ablation(None)
    c = cases(None)
    torch.cuda.synchronize(); t = time.time()
    for x in c: q.score_many(*x, fast=False)
    torch.cuda.synchronize(); slow = (time.time() - t) / (len(c) * len(GATE_CONDS))
    t = time.time()
    for x in c: q.score_many(*x, fast=True)
    torch.cuda.synchronize(); fast = (time.time() - t) / (len(c) * len(GATE_CONDS))
    res['sec_per_prompt_slow'], res['sec_per_prompt_fast'] = round(slow, 4), round(fast, 4)
    res['ok'] = all(res[k]['ok'] for k in ('base', 'constitution', 'ablation_random_dir'))
    (HERE / 'results/fast_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
