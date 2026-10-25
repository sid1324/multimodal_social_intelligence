"""Bridges-2: minimal GRPO with LoRA on Qwen3-VL-8B-Instruct for EgoNormia (arms R0-R3, rewards in rl_common.py).

Per step: P prompts, G sampled completions each (temperature 1.0, full vocabulary, max 384 new tokens). Advantage
A = (r - mean_group) / (std_group + 1e-4). Single on-policy update per batch, so the PPO ratio is 1 and the loss is
  L = mean_seq [ mean_tok( -A * exp(logp - sg(logp)) + beta * (exp(ref - logp) - (ref - logp) - 1) ) ]
with ref = the base model (adapter disabled). Groups whose rewards are all equal carry no policy gradient and are
skipped (logged as frac_flat). Options are reshuffled every epoch from a per-(epoch, item) seed.
usage: grpo_train.py --arm R1 --out adapters/R1_s0 --steps 90 [--seed 0] [--lr 1e-5] [--beta 0.04]
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import rl_common as rc

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
TARGETS = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'


def shuffled(item, epoch, seed):
    rng = random.Random(f'{seed}:{epoch}:{item["id"]}')
    order = {}
    for kind in ('action', 'justification'):
        perm = list(range(5))
        rng.shuffle(perm)
        order[kind] = [item['base_order'][kind][p] for p in perm]
    return order


def conversation(item, order, fmt):
    text = rc.prompt(fmt, [item['behaviors'][i] for i in order['action']], [item['justifications'][i] for i in order['justification']])
    image = Image.open(ROOT / 'data/train_images' / item['image']).convert('RGB')
    return [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': text}]}]


def completion_mask(comp, eos_ids):
    is_eos = torch.zeros_like(comp, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= comp == e
    before = torch.cumsum(is_eos.int(), dim=1) - is_eos.int()  # number of EOS strictly before t
    return (before == 0).long()


def token_logps(model, ids, mask, pixel_values, grid, n_comp):
    out = model(input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=grid, logits_to_keep=n_comp + 1)
    logits = out.logits[:, :n_comp].float()
    return torch.gather(logits.log_softmax(-1), 2, ids[:, -n_comp:].unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(rc.ARMS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prompts', type=int, default=8)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=0.04)
    ap.add_argument('--max-new', type=int, default=384)
    ap.add_argument('--micro', type=int, default=4)
    ap.add_argument('--gen-chunk', type=int, default=2)
    ap.add_argument('--save-every', type=int, default=10)
    args = ap.parse_args()
    fmt = rc.ARMS[args.arm]
    torch.manual_seed(args.seed)
    items = json.loads((ROOT / 'data/train_items.json').read_text())
    labels = json.loads((ROOT / 'data/train_spatial_labels.json').read_text()) if fmt == 'spatial' else {}
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = 'left'
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, task_type='CAUSAL_LM'))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    tok = processor.tokenizer
    eos_ids = sorted({tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id, tok.pad_token_id} - {None})

    schedule, epoch = [], 0
    while len(schedule) < args.steps * args.prompts:
        order = list(range(len(items)))
        random.Random(f'{args.seed}:order:{epoch}').shuffle(order)
        schedule += [(epoch, i) for i in order]
        epoch += 1

    log_path = out_dir / 'train_log.jsonl'
    for step in range(args.steps):
        t0 = time.time()
        batch = schedule[step * args.prompts:(step + 1) * args.prompts]
        picked = [(items[i], shuffled(items[i], ep, args.seed)) for ep, i in batch]
        convs = [conversation(item, order, fmt) for item, order in picked]
        inputs = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors='pt', padding=True).to('cuda')
        # Generation: no checkpointing, KV cache on.
        model.eval()
        model.base_model.model.gradient_checkpointing_disable()
        # Sample a few prompts at a time: 64 long spatial completions at once ran out of memory on a 48 GB L40S.
        sizes = inputs['image_grid_thw'].prod(dim=1).tolist()
        starts = [sum(sizes[:p]) for p in range(len(sizes))]
        chunks = []
        with torch.no_grad():
            for c in range(0, args.prompts, args.gen_chunk):
                ps = list(range(c, min(c + args.gen_chunk, args.prompts)))
                sub = {'input_ids': inputs['input_ids'][ps], 'attention_mask': inputs['attention_mask'][ps],
                       'pixel_values': torch.cat([inputs['pixel_values'][starts[p]:starts[p] + sizes[p]] for p in ps]),
                       'image_grid_thw': inputs['image_grid_thw'][ps]}
                chunks.append(model.generate(**sub, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                             max_new_tokens=args.max_new, num_return_sequences=args.group, use_cache=True))
        width = max(s.shape[1] for s in chunks)
        seqs = torch.cat([torch.nn.functional.pad(s, (0, width - s.shape[1]), value=tok.pad_token_id) for s in chunks])
        del chunks
        torch.cuda.empty_cache()
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        t_gen = time.time() - t0
        prompt_len = inputs['input_ids'].shape[1]
        comp = seqs[:, prompt_len:]
        cmask = completion_mask(comp, eos_ids)
        texts = tok.batch_decode(comp, skip_special_tokens=True)

        rewards, parts = [], []
        for r, text in enumerate(texts):
            item, order = picked[r // args.group]
            total, info = rc.reward(args.arm, text, order, item['gold'], labels.get(item['id']))
            rewards.append(total)
            parts.append(info)
        rew = torch.tensor(rewards).view(args.prompts, args.group)
        adv = ((rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-4)).view(-1)
        active = [g for g in range(args.prompts) if rew[g].std() > 1e-6]

        # Policy update over the rows of non-flat groups, in micro-batches.
        model.train()
        # Images differ in size, so each prompt's patches are sliced by its own grid (sizes/starts above).
        rows = [r for g in active for r in range(g * args.group, (g + 1) * args.group)]
        kl_sum, n_rows = 0.0, len(rows)
        for s in range(0, len(rows), args.micro):
            idx = torch.tensor(rows[s:s + args.micro], device='cuda')
            pidx = idx // args.group
            ids = seqs[idx]
            mask = torch.cat([inputs['attention_mask'][pidx], cmask[idx]], dim=1)
            pv = torch.cat([inputs['pixel_values'][starts[p]:starts[p] + sizes[p]] for p in pidx.tolist()])
            grid = inputs['image_grid_thw'][pidx]
            n_comp = comp.shape[1]
            with torch.no_grad(), model.disable_adapter():
                ref = token_logps(model, ids, mask, pv, grid, n_comp)
            logp = token_logps(model, ids, mask, pv, grid, n_comp)
            m = cmask[idx].float()
            delta = ref - logp
            kl = torch.exp(delta) - delta - 1
            a = adv[idx.cpu()].to('cuda').unsqueeze(1)
            per_tok = -a * torch.exp(logp - logp.detach()) + args.beta * kl
            loss = ((per_tok * m).sum(1) / m.sum(1).clamp(min=1)).sum() / max(1, n_rows)
            loss.backward()
            kl_sum += float(((kl * m).sum(1) / m.sum(1).clamp(min=1)).sum())
        if rows:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        rec = {'step': step + 1, 'reward': float(rew.mean()), 'r_norm': sum(p['r_norm'] for p in parts) / len(parts),
               'r_spatial': sum(p['r_spatial'] for p in parts) / len(parts), 'parsed': sum(p['parsed'] for p in parts) / len(parts),
               'joint': sum(p['action'] and p['justification'] for p in parts) / len(parts),
               'len': float(cmask.sum(1).float().mean()), 'kl': kl_sum / max(1, n_rows), 'frac_flat': 1 - len(active) / args.prompts,
               'sec_gen': round(t_gen, 1), 'sec': round(time.time() - t0, 1)}
        with log_path.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
        if step == 0:
            (out_dir / 'sample_completions.json').write_text(json.dumps(texts[:args.group], indent=1))
        stop = (out_dir / 'STOP').exists()  # touch adapters/<name>/STOP to end early with the adapter saved
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps or stop:
            model.save_pretrained(out_dir)
            (out_dir / 'state.json').write_text(json.dumps({'step': step + 1, 'args': vars(args), 'stopped_early': stop}))
        if stop:
            break
    print('TRAIN_DONE', args.arm, args.out)


if __name__ == '__main__':
    main()
