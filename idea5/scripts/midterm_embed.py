"""One Qwen3-VL-Embedding-8B vector per (item, input condition), plus one per answer option.

Each image is followed by its own description in a single interleaved sequence, pooled at the last token and
L2-normalised (the model's own recipe). Conditions (PLAN.md section 3):
  C0 null text | C1 RGB | C2 RGB + D_rgb | C3 C2 + depth/seg/pose images | C4 C2 + mask descriptions
  C5 C2 + each mask image followed by its description | C6 mask descriptions only
  X1 C5 with masks and descriptions from another item | X2 C2 + non-spatial filler of the mask-description length
  X3 C2 + the RGB strip three more times
Output: output/idea5-midterm/emb/<cond>/<id>.npy (float16). Resumable.

usage: midterm_embed.py --conds C1,C2,... [--split test,train] [--limit N] | --options
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils.vision_process import process_vision_info
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/Qwen3-VL-Embedding-8B'
sys.path.insert(0, str(MODEL / 'scripts'))
from qwen3_vl_embedding import Qwen3VLForEmbedding  # noqa: E402

DATA = ROOT / 'data/EgoNormia'
OUT = ROOT / 'output/idea5-midterm'
GRIDS = OUT / 'grids'
DESC = OUT / 'descriptions'
EMB = OUT / 'emb'
MIN_PIXELS, MAX_PIXELS = 4 * 32 * 32, 800 * 32 * 32
CONTEXT_INSTRUCTION = 'Represent this egocentric scene for choosing the socially appropriate next action.'
OPTION_INSTRUCTION = {'action': 'Represent this candidate action.', 'justification': 'Represent this candidate justification.'}
NONE_TEXT = {'action': 'None of these actions is appropriate.', 'justification': 'None of these justifications is appropriate.'}
LABEL = {'rgb': 'Video frames', 'depth': 'Depth map (brighter is nearer)', 'segmentation': 'Instance segmentation map',
         'pose': 'Body and hand skeleton map'}
MASKS = ['depth', 'segmentation', 'pose']
# Neutral passage with no spatial or social content, cut to length for X2.
FILLER = ('The recipe for a basic bread dough has changed little over centuries. Flour, water, salt and yeast are combined '
          'and left to ferment, and the length of fermentation affects flavour more than any other single choice. '
          'Bakers record hydration as a percentage of flour weight, and most everyday loaves sit between sixty and '
          'seventy-five percent. Higher hydration gives a more open crumb but a stickier dough that is harder to shape. '
          'Salt is usually about two percent of the flour weight; it slows fermentation slightly and tightens gluten. '
          'Whole grain flours absorb more water than white flour, so recipes that swap one for the other often adjust '
          'the water as well. Sourdough replaces commercial yeast with a culture of wild yeast and lactic acid bacteria, '
          'which adds acidity and keeps the bread from staling as quickly. ') * 6


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def image_part(item_id, kind):
    return {'type': 'image', 'image': 'file://' + str(GRIDS / f'{item_id}_{kind}.jpg'),
            'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS}


def text_part(text):
    return {'type': 'text', 'text': text}


def descriptions(item_id):
    rgb = json.loads((DESC / 'rgb' / f'{item_id}.json').read_text())['descriptions']
    maps = json.loads((DESC / 'maps' / f'{item_id}.json').read_text())['descriptions']
    return {**rgb, **maps}


class Embedder:
    def __init__(self, device=None):
        self.device = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
        # bf16 + SDPA on MPS gives wrong embeddings (fails the model-card similarity check); bf16 + eager matches CPU.
        self.model = Qwen3VLForEmbedding.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,
                                                         attn_implementation='eager').to(self.device).eval()
        self.processor = Qwen3VLProcessor.from_pretrained(str(MODEL), padding_side='right')

    @torch.no_grad()
    def __call__(self, contents, instruction):
        conversations = [[{'role': 'system', 'content': [text_part(instruction)]}, {'role': 'user', 'content': c}] for c in contents]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, _, _ = process_vision_info(conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
        inputs = self.processor(text=text, images=images, truncation=False, padding=True, do_resize=False, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs['attention_mask']
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
        vectors = F.normalize(hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float(), p=2, dim=-1)
        vectors, tokens = vectors.cpu().numpy(), int(mask.sum())
        del inputs, hidden
        if self.device.type == 'mps':
            torch.mps.empty_cache()  # cached blocks otherwise grow until MPS runs out of memory
        return vectors, tokens


def partner(item_id, pool):
    """Fixed other item from a different source video, for the mismatched-map control."""
    source = item_id.split('_')[0]
    candidates = sorted((i for i in pool if i.split('_')[0] != source), key=lambda i: digest(f'x1:{item_id}:{i}'))
    return candidates[0]


def content(cond, item_id, pool, tokenizer):
    if cond == 'C0':
        return [text_part('No visual input.')]
    if cond == 'C1':
        return [image_part(item_id, 'rgb')]
    d = descriptions(item_id)
    base = [image_part(item_id, 'rgb'), text_part(f"{LABEL['rgb']}: {d['rgb']}")]
    if cond == 'C2':
        return base
    if cond == 'C3':
        return base + [image_part(item_id, k) for k in MASKS]
    if cond == 'C4':
        return base + [text_part(f'{LABEL[k]}: {d[k]}') for k in MASKS]
    if cond == 'C5':
        return base + [p for k in MASKS for p in (image_part(item_id, k), text_part(f'{LABEL[k]}: {d[k]}'))]
    if cond == 'C6':
        return [text_part(' '.join(f'{LABEL[k]}: {d[k]}' for k in MASKS))]
    if cond == 'X1':
        other = partner(item_id, pool)
        od = descriptions(other)
        return base + [p for k in MASKS for p in (image_part(other, k), text_part(f'{LABEL[k]}: {od[k]}'))]
    if cond == 'X2':
        n = len(tokenizer(' '.join(f'{LABEL[k]}: {d[k]}' for k in MASKS))['input_ids'])
        filler = tokenizer.decode(tokenizer(FILLER)['input_ids'][:n])
        return base + [text_part(filler)]
    if cond == 'X3':
        return base + [image_part(item_id, 'rgb') for _ in range(3)]
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conds', default='')
    ap.add_argument('--split', default='test,train')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--options', action='store_true')
    ap.add_argument('--max-forwards', type=int, default=150,
                    help='exit after this many forwards; MPS graph caches for new shapes grow without bound')
    args = ap.parse_args()
    split = json.loads((OUT / 'splits.json').read_text())
    ids = [i for s in args.split.split(',') for i in split[s]][:args.limit]
    pool = [i for s in ('test', 'train', 'train_ext') for i in split[s]]
    emb = Embedder()
    log = EMB / 'manifest.jsonl'
    EMB.mkdir(parents=True, exist_ok=True)

    if args.options:
        data = json.loads((DATA / 'annotations/final_data.json').read_text())
        for head, field in (('action', 'behaviors'), ('justification', 'justifications')):
            dest = EMB / f'options_{head}'
            dest.mkdir(exist_ok=True)
            todo = [i for i in ids if not (dest / f'{i}.npy').exists()]
            for item_id in todo:
                texts = [t if t else NONE_TEXT[head] for t in data[item_id][field]]
                vectors, _ = emb([[text_part(t)] for t in texts], OPTION_INSTRUCTION[head])
                np.save(dest / f'{item_id}.npy', vectors.astype(np.float16))
            print('options', head, len(todo), flush=True)

    forwards = 0
    for cond in [c for c in args.conds.split(',') if c]:
        dest = EMB / cond
        dest.mkdir(exist_ok=True)
        done = 0
        for item_id in ids:
            if (dest / f'{item_id}.npy').exists():
                continue
            try:
                parts = content(cond, item_id, pool, emb.processor.tokenizer)
            except FileNotFoundError:
                continue  # descriptions not ready yet; a later run picks this item up
            t = time.time()
            try:
                vectors, tokens = emb([parts], CONTEXT_INSTRUCTION)
            except (ValueError, RuntimeError) as e:
                with (EMB / 'errors.jsonl').open('a') as f:
                    f.write(json.dumps({'cond': cond, 'id': item_id, 'error': str(e)[:300]}) + '\n')
                if 'out of memory' in str(e):
                    print('FORWARD_BUDGET_REACHED (out of memory; restarting)', flush=True)
                    return  # the MPS allocator does not recover in-process; the driver starts a fresh one
                continue
            if not np.isfinite(vectors[0]).all():
                with (EMB / 'errors.jsonl').open('a') as f:
                    f.write(json.dumps({'cond': cond, 'id': item_id, 'error': 'non-finite vector'}) + '\n')
                continue
            np.save(dest / f'{item_id}.npy', vectors[0].astype(np.float16))
            forwards += 1
            with log.open('a') as f:
                f.write(json.dumps({'cond': cond, 'id': item_id, 'tokens': tokens, 'seconds': round(time.time() - t, 2)}) + '\n')
            done += 1
            if done % 50 == 0:
                print(cond, done, flush=True)
            if forwards >= args.max_forwards:
                break
        print('done', cond, done, flush=True)
        if forwards >= args.max_forwards:
            print('FORWARD_BUDGET_REACHED', flush=True)
            return


if __name__ == '__main__':
    main()
