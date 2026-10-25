"""Runpod: LoRA fine-tune Qwen3-VL-Embedding-8B end to end on EgoNormia, one input condition per run.

Context sequences are built exactly as in scripts/midterm_embed.py (content(), partner(), descriptions(), same
instructions, pixel limits, last-token pooling + L2 norm). Options are embedded by the same LoRA model. Per item:
  logits_a = c . o_a / tau,  logits_j = c . o_j / tau  (5 options each, empty option = NONE_TEXT)
  L = CE(logits_a, gold) + CE(logits_j, gold)
LoRA r=16 alpha=32 on language-model q/k/v/o/gate/up/down (vision tower frozen), trained on train + train_ext (600),
AdamW, gradient accumulation over items. Test labels are not on this machine: test predictions and all option scores
are written for local scoring.

usage: emb_ft.py --zeroshot --conds C2,C5,X1 --out preds/zeroshot
       emb_ft.py --cond C5 --seed 0 --out runs/C5_s0 [--epochs 2] [--limit 8]
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from qwen_vl_utils.vision_process import process_vision_info
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'Qwen3-VL-Embedding-8B'
sys.path.insert(0, str(ROOT))
from qwen3_vl_embedding import Qwen3VLForEmbedding  # noqa: E402

GRIDS = ROOT / 'grids'
DESC = ROOT / 'descriptions'
MIN_PIXELS, MAX_PIXELS = 4 * 32 * 32, 800 * 32 * 32
CONTEXT_INSTRUCTION = 'Represent this egocentric scene for choosing the socially appropriate next action.'
OPTION_INSTRUCTION = {'action': 'Represent this candidate action.', 'justification': 'Represent this candidate justification.'}
NONE_TEXT = {'action': 'None of these actions is appropriate.', 'justification': 'None of these justifications is appropriate.'}
LABEL = {'rgb': 'Video frames', 'depth': 'Depth map (brighter is nearer)', 'segmentation': 'Instance segmentation map',
         'pose': 'Body and hand skeleton map'}
MASKS = ['depth', 'segmentation', 'pose']
TARGETS = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'


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


def partner(item_id, pool):
    """Fixed other item from a different source video, for the mismatched-map control."""
    source = item_id.split('_')[0]
    candidates = sorted((i for i in pool if i.split('_')[0] != source), key=lambda i: digest(f'x1:{item_id}:{i}'))
    return candidates[0]


def content(cond, item_id, pool):
    d = descriptions(item_id)
    base = [image_part(item_id, 'rgb'), text_part(f"{LABEL['rgb']}: {d['rgb']}")]
    if cond == 'C2':
        return base
    if cond == 'C5':
        return base + [p for k in MASKS for p in (image_part(item_id, k), text_part(f'{LABEL[k]}: {d[k]}'))]
    if cond == 'X1':
        other = partner(item_id, pool)
        od = descriptions(other)
        return base + [p for k in MASKS for p in (image_part(other, k), text_part(f'{LABEL[k]}: {od[k]}'))]
    raise ValueError(cond)


class Embedder:
    def __init__(self, lora_seed=None):
        self.model = Qwen3VLForEmbedding.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,
                                                         attn_implementation='sdpa').to('cuda')
        self.processor = Qwen3VLProcessor.from_pretrained(str(MODEL), padding_side='right')
        if lora_seed is not None:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.model.enable_input_require_grads()
            torch.manual_seed(lora_seed)
            self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS))
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()  # fp32 adapter weights; peft casts the LoRA output back to bf16

    def __call__(self, contents, instruction):
        conversations = [[{'role': 'system', 'content': [text_part(instruction)]}, {'role': 'user', 'content': c}] for c in contents]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, _, _ = process_vision_info(conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
        inputs = self.processor(text=text, images=images, truncation=False, padding=True, do_resize=False, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        hidden = self.model(**inputs, use_cache=False).last_hidden_state
        mask = inputs['attention_mask']
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
        return F.normalize(hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float(), p=2, dim=-1), int(mask.sum())


def option_texts(item, head):
    return [t if t else NONE_TEXT[head] for t in item['behaviors' if head == 'action' else 'justifications']]


def score_item(emb, cond, item_id, item, pool):
    c, tokens = emb([content(cond, item_id, pool)], CONTEXT_INSTRUCTION)
    out = {}
    for head in ('action', 'justification'):
        o, _ = emb([[text_part(t)] for t in option_texts(item, head)], OPTION_INSTRUCTION[head])
        out[head] = (c @ o.T)[0]
    return out, tokens


@torch.no_grad()
def predict(emb, cond, ids, items, pool):
    rows = {}
    for item_id in ids:
        try:
            s, _ = score_item(emb, cond, item_id, items[item_id], pool)
        except FileNotFoundError:
            continue
        rows[item_id] = {head: [round(float(x), 6) for x in s[head]] for head in s}
        rows[item_id]['empty_action'] = [int(not t) for t in items[item_id]['behaviors']]
        rows[item_id]['empty_justification'] = [int(not t) for t in items[item_id]['justifications']]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zeroshot', action='store_true')
    ap.add_argument('--conds', default='C2,C5,X1')
    ap.add_argument('--cond')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    meta = json.loads((ROOT / 'items.json').read_text())
    items, split = meta['items'], meta['splits']
    pool = split['test'] + split['train'] + split['train_ext']
    test_ids = split['test'][:args.limit]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.zeroshot:
        emb = Embedder()
        emb.model.eval()
        for cond in args.conds.split(','):
            t0 = time.time()
            rows = predict(emb, cond, test_ids, items, pool)
            (out / f'{cond}.json').write_text(json.dumps({'cond': cond, 'mode': 'zeroshot', 'rows': rows}))
            print('ZS_DONE', cond, len(rows), round(time.time() - t0), 's', flush=True)
        return

    emb = Embedder(lora_seed=args.seed)
    params = [p for p in emb.model.parameters() if p.requires_grad]
    print('trainable', sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    train_ids = [i for i in split['train'] + split['train_ext'] if (DESC / 'rgb' / f'{i}.json').exists()][:args.limit]
    log = (out / 'train_log.jsonl').open('a')
    step, acc_loss, acc_hits, acc_n, t0 = 0, 0.0, 0, 0, time.time()
    emb.model.train()
    for epoch in range(args.epochs):
        order = list(train_ids)
        random.Random(f'{args.seed}:{epoch}').shuffle(order)
        for k, item_id in enumerate(order):
            item = items[item_id]
            try:
                s, tokens = score_item(emb, args.cond, item_id, item, pool)
            except FileNotFoundError:
                continue
            gold = torch.tensor([item['correct']], device='cuda')
            loss = sum(F.cross_entropy(s[h][None] / args.tau, gold) for h in ('action', 'justification'))
            (loss / args.accum).backward()
            acc_loss += float(loss.detach())
            acc_hits += int(int(s['action'].argmax()) == item['correct'] and int(s['justification'].argmax()) == item['correct'])
            acc_n += 1
            if acc_n == args.accum or k == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {'step': step, 'epoch': epoch, 'loss': acc_loss / acc_n, 'train_joint': acc_hits / acc_n, 'tokens': tokens,
                       'sec_per_item': round((time.time() - t0) / acc_n, 2), 'mem_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                log.write(json.dumps(rec) + '\n')
                log.flush()
                if step % 5 == 0 or args.limit:
                    print(json.dumps(rec), flush=True)
                acc_loss, acc_hits, acc_n, t0 = 0.0, 0, 0, time.time()
    emb.model.save_pretrained(out / 'adapter')
    emb.model.eval()
    t1 = time.time()
    rows = predict(emb, args.cond, test_ids, items, pool)
    (out / 'test_preds.json').write_text(json.dumps({'cond': args.cond, 'seed': args.seed, 'mode': 'lora', 'args': vars(args), 'rows': rows}))
    print('RUN_DONE', args.cond, args.seed, len(rows), round(time.time() - t1), 's eval', flush=True)


if __name__ == '__main__':
    main()
