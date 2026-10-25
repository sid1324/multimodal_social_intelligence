"""Qwen3-VL wrapper: loading, option-letter scoring, residual capture and directional ablation.

Residual stream = the output of each decoder layer (model.model.language_model.layers[l]), which in
transformers 4.57 is a plain tensor. DeepStack adds visual features after layers 0-2, outside these hooks.
Ablation: h <- h - alpha * (h . u) u at every position, for the configured layers.
"""
import os, time
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


def pick_dtype():
    """bf16 where the GPU supports it (H100/L40S/A100), else fp16 (V100)."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


class Qwen:
    def __init__(self, dtype=None, device='cuda'):
        self.dtype = dtype or pick_dtype()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=True, dtype=self.dtype,
            attn_implementation='sdpa', device_map=device).eval()
        self.layers = self.model.model.language_model.layers
        self.n_layers = len(self.layers)
        tok = self.processor.tokenizer
        self.letter_ids = {}
        for L in 'ABCDE':
            ids = tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, (L, ids)
            self.letter_ids[L] = ids[0]
        self._capture = None      # list to receive residuals at self._capture_idx, or None
        self._capture_idx = None  # LongTensor [B]: position to capture in each row
        self._ablate = {}         # layer -> (unit vector [H] on device, alpha)
        self._hooks = [layer.register_forward_hook(self._make_hook(l)) for l, layer in enumerate(self.layers)]

    # ---- hooks -------------------------------------------------------------------------------
    def _make_hook(self, l):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if l in self._ablate:
                u, alpha = self._ablate[l]
                hf = h.float()
                hf = hf - alpha * (hf @ u).unsqueeze(-1) * u
                h = hf.to(h.dtype)
            if self._capture is not None:
                rows = torch.arange(h.shape[0], device=h.device)
                self._capture.append(h[rows, self._capture_idx.to(h.device)].detach().float().cpu())
            if l in self._ablate:
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def set_ablation(self, directions=None, alpha=1.0, layers=()):
        """directions: array [n_layers, H] (unit-normalized here) or None to clear."""
        self._ablate = {}
        if directions is None:
            return
        for l in layers:
            u = torch.as_tensor(np.asarray(directions[l]), dtype=torch.float32, device=self.device)
            self._ablate[l] = (u / u.norm(), float(alpha))

    # ---- inputs ------------------------------------------------------------------------------
    def inputs(self, images, text, system=None):
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': text}]
        msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': system}]}] if system else []) + \
               [{'role': 'user', 'content': content}]
        chat = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[chat], images=list(images), return_tensors='pt')
        return enc.to(self.device), chat

    # ---- scoring -----------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, images, text, n_opts, system=None, capture=False):
        """Next-token log-probs over the first n_opts letters.
        Returns dict: letter_logp (log-softmax over valid letters), letter_mass (prob mass the full
        vocabulary puts on valid letters), argmax letter, finite flag, token counts, optional residuals."""
        enc, _ = self.inputs(images, text, system)
        self._capture = [] if capture else None
        self._capture_idx = torch.tensor([enc['input_ids'].shape[1] - 1])
        logits = self.model(**enc, logits_to_keep=1).logits[0, -1].float()
        resid = torch.stack(self._capture)[:, 0].half().numpy() if capture else None
        self._capture = None
        return self._letters(logits, n_opts, enc['input_ids'].shape[1], enc.get('image_grid_thw')), resid

    def _letters(self, logits, n_opts, n_tokens, grid):
        letters = 'ABCDE'[:n_opts]
        ids = torch.tensor([self.letter_ids[L] for L in letters], device=logits.device)
        full = torch.log_softmax(logits, -1)
        sel = logits[ids]
        lp = torch.log_softmax(sel, -1)
        return {'letter_logp': lp.tolist(), 'argmax': letters[int(sel.argmax())],
                'letter_mass': float(full[ids].exp().sum()), 'finite': bool(torch.isfinite(logits).all()),
                'n_tokens': int(n_tokens), 'image_grid_thw': grid.tolist() if grid is not None else None}

    @torch.no_grad()
    def score_many(self, images, texts, n_opts, system=None, capture=False, fast=False):
        """Score several prompts that share the same images (e.g. all cue conditions of one item).
        fast=False: one full forward per prompt (reference path).
        fast=True: run the shared prefix (system + images + identical leading text) once with the
        current hooks, then all differing suffixes as one right-padded batch on a copy of its KV cache.
        Only valid if verify_fast() shows it matches the reference path."""
        if not fast:
            outs = [self.score(images, t, n_opts, system, capture) for t in texts]
            return [o for o, _ in outs], ([r for _, r in outs] if capture else None)
        tok = self.processor.tokenizer
        encs = [self.inputs(images, t, system)[0] for t in texts[:1]]
        ids0 = encs[0]['input_ids'][0]
        # expand the other prompts' image placeholders exactly as the processor did for prompt 0
        chats = []
        for t in texts:
            content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': t}]
            msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': system}]}] if system else []) + \
                   [{'role': 'user', 'content': content}]
            chats.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        pad_id = tok.convert_tokens_to_ids('<|image_pad|>')
        counts = [int(n) for n in (encs[0]['image_grid_thw'].prod(-1) // (self.processor.image_processor.merge_size ** 2)).tolist()]
        def expand(chat):
            parts = chat.split('<|image_pad|>')
            assert len(parts) == len(counts) + 1
            return ''.join(p + ('<|image_pad|>' * c if i < len(counts) else '') for i, (p, c) in enumerate(zip(parts, counts + [0])))
        all_ids = [torch.tensor(tok(expand(c), add_special_tokens=False)['input_ids']) for c in chats]
        assert torch.equal(all_ids[0], ids0.cpu()), 'manual image expansion differs from the processor'
        P = min(len(x) for x in all_ids)
        for x in all_ids[1:]:
            neq = (x[:P] != all_ids[0][:P]).nonzero()
            if len(neq):
                P = min(P, int(neq[0]))
        last_img = int((ids0 == pad_id).nonzero().max())
        assert last_img < P < min(len(x) for x in all_ids), (last_img, P)
        dev = self.device
        # prefix (hooks active, no capture)
        self._capture = None
        out = self.model(input_ids=ids0[None, :P].to(dev), attention_mask=torch.ones(1, P, dtype=torch.long, device=dev),
                         pixel_values=encs[0]['pixel_values'], image_grid_thw=encs[0]['image_grid_thw'], use_cache=True,
                         logits_to_keep=1)
        cache = out.past_key_values
        rope_deltas = self.model.model.rope_deltas
        B, lens = len(texts), [len(x) - P for x in all_ids]
        L = max(lens)
        suf = torch.full((B, L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros(B, P + L, dtype=torch.long)
        mask[:, :P] = 1
        for b, x in enumerate(all_ids):
            suf[b, :lens[b]] = x[P:]
            mask[b, P:P + lens[b]] = 1
        cache.batch_repeat_interleave(B)
        pos = (torch.arange(P, P + L, device=dev)[None, :] + rope_deltas.to(dev)).expand(B, -1)
        pos = pos[None].expand(3, -1, -1)
        last = torch.tensor([l - 1 for l in lens])
        self._capture = [] if capture else None
        self._capture_idx = last
        logits = self.model(input_ids=suf.to(dev), attention_mask=mask.to(dev), position_ids=pos,
                            past_key_values=cache, cache_position=torch.arange(P, P + L, device=dev),
                            use_cache=True).logits.float()
        resid = torch.stack(self._capture, 1).half().numpy() if capture else None  # [B, n_layers, H]
        self._capture = None
        outs = [self._letters(logits[b, lens[b] - 1], n_opts, len(all_ids[b]), encs[0]['image_grid_thw']) for b in range(B)]
        return outs, ([resid[b] for b in range(B)] if capture else None)

    def verify_fast(self, cases, tol=0.05):
        """cases: list of (images, texts, n_opts, system). Compares fast vs reference letter log-probs
        and captured residuals. Returns a summary dict; 'ok' requires identical argmax everywhere and
        max |delta logp| <= tol."""
        dmax, agree, n, cos_min = 0.0, 0, 0, 1.0
        for images, texts, n_opts, system in cases:
            ref, rr = self.score_many(images, texts, n_opts, system, capture=True, fast=False)
            fst, rf = self.score_many(images, texts, n_opts, system, capture=True, fast=True)
            for a, b, x, y in zip(ref, fst, rr, rf):
                dmax = max(dmax, max(abs(p - q) for p, q in zip(a['letter_logp'], b['letter_logp'])))
                agree += a['argmax'] == b['argmax']
                n += 1
                xa, ya = torch.tensor(x[-1]).float(), torch.tensor(y[-1]).float()
                cos_min = min(cos_min, float(torch.nn.functional.cosine_similarity(xa, ya, dim=0)))
        return {'n_prompts': n, 'argmax_agree': agree / n, 'max_abs_dlogp': dmax, 'min_cos_last_layer_resid': cos_min,
                'ok': agree == n and dmax <= tol}

    @torch.no_grad()
    def greedy(self, images, text, system=None, max_new_tokens=4):
        enc, _ = self.inputs(images, text, system)
        gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.tokenizer.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
