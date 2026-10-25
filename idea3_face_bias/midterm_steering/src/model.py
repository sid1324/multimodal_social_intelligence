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
        self._capture = None      # list to receive last-token residuals, or None
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
                self._capture.append(h[0, -1].detach().float().cpu())
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
        logits = self.model(**enc, logits_to_keep=1).logits[0, -1].float()
        resid = torch.stack(self._capture).half().numpy() if capture else None
        self._capture = None
        letters = 'ABCDE'[:n_opts]
        ids = torch.tensor([self.letter_ids[L] for L in letters], device=logits.device)
        full = torch.log_softmax(logits, -1)
        sel = logits[ids]
        lp = torch.log_softmax(sel, -1)
        out = {'letter_logp': lp.tolist(), 'argmax': letters[int(sel.argmax())],
               'letter_mass': float(full[ids].exp().sum()), 'finite': bool(torch.isfinite(logits).all()),
               'n_tokens': int(enc['input_ids'].shape[1]),
               'image_grid_thw': enc['image_grid_thw'].tolist() if 'image_grid_thw' in enc else None}
        return out, resid

    @torch.no_grad()
    def greedy(self, images, text, system=None, max_new_tokens=4):
        enc, _ = self.inputs(images, text, system)
        gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.tokenizer.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
