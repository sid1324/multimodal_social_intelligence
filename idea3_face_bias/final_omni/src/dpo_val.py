"""Validation scoring of one method on one cell set (Nov 25; resumable, one file per method and set so lanes can work in parallel).
Validation split only. Sets: text2 = L2 text plus clean accuracy items, text1 = L1 text, image2 / speech2 = L2 image / speech.
The L2 text numbers are the pre-registered selection measure; L1 and image are descriptive here.
Usage: python src/dpo_val.py METHOD SET     METHOD: base or dpo_text_e1, dpo_multi_e2, ..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, load_splits
from evaluate import run_grid, level_conds


def cells(st):
    if st == 'text2':
        return [('none', 'text')] + [(c, 'text') for c in level_conds('L2')]
    lv, ch = {'text1': ('L1', 'text'), 'image2': ('L2', 'image'), 'speech2': ('L2', 'speech')}[st]
    return [(c, ch) for c in level_conds(lv)]


def score(m, method, st):
    sp = load_splits(); pool = sp['pool']['val']
    out = HERE / f'results/dpo/nemotron_val_{method}_{st}.jsonl'
    ok = run_grid(m, out, method, pool, cells(st), 'val')
    if ok and st == 'text2':
        ok = run_grid(m, out, method, [k for k in sp['clean']['val'] if k not in set(pool)], [('none', 'text')], 'val')
    return ok


def main():
    method, st = sys.argv[1], sys.argv[2]
    from omni_model import load
    m = load('nemotron')
    if method != 'base':
        from dpo_train import adapter_dirs
        from peft import PeftModel
        path = dict(adapter_dirs('nemotron'))[method]
        pm = PeftModel.from_pretrained(m.lm, path, adapter_name='x'); pm.set_adapter('x'); pm.eval()
        n = sum(p.abs().sum().item() > 0 for k, p in pm.named_parameters() if 'lora_B' in k)
        assert n > 0, 'adapter has no nonzero lora_B'
        print('adapter', path, 'nonzero lora_B tensors', n, flush=True)
    sys.exit(0 if score(m, method, st) else 75)


if __name__ == '__main__':
    main()
