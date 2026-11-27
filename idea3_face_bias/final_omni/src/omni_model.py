"""Omni-model wrapper (port of midterm model.py): one interface, two backends (Nemotron 3 Nano Omni, Qwen3-Omni).

A prompt is a list of parts [('image', PIL.Image), ('text', str), ('audio', wav path)], plus an optional system text.
Scoring: next-token log-probs over the valid option letters, with a float32 copy of the output head (midterm D12).
Residual stream = output of every language-model block. For Nemotron's hybrid stack this covers Mamba2, attention
and MoE blocks alike (each is its own pre-norm residual block), so "layer" is a position in the residual stream.
Ablation: h <- h - alpha * (h . u) u at every position of the configured layers.

Batching: Nemotron's Mamba cache advances one token at a time, so the midterm's KV-prefix path cannot be used.
Instead the image (and audio) embeddings are computed once and reused, and prompts are right-padded into one batch
without an attention mask; every block is causal or per-token, so right padding cannot change earlier positions.
"""
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch

REV = {'nemotron': ('nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16', 'e5e9932441de940c9a62185c870ea5bcd4cd24e2'),
       'qwen': ('Qwen/Qwen3-Omni-30B-A3B-Instruct', '26291f793822fb6be9555850f06dfe95f2d7e695'),
       'gemma': ('google/gemma-4-12B-it', '43c9d014b231474fa7212dbb48a762566c2b255d')}
MAX_BATCH_TOKENS = int(os.environ.get('MAX_BATCH_TOKENS', 16000))


def snapshot(name):
    mid, rev = REV[name]
    fast = Path('/work/nvme/bjuv/ylothe/idea3_weights') / name
    if (fast / '.done').exists():      # NVMe copy of the same pinned snapshot (the HDD copy takes 11 min to load)
        return str(fast / rev)
    return str(Path(os.environ['HF_HOME']) / 'hub' / ('models--' + mid.replace('/', '--')) / 'snapshots' / rev)


def pin_kernels():
    """Pinned prebuilt CUDA kernels (the model card code warns that its pure-PyTorch Mamba fallback degrades accuracy).
    A commit revision also resolves offline, unlike the branch lookup behind 'version'."""
    from transformers.integrations import hub_kernels as hk
    hk._HUB_KERNEL_MAPPING['mamba-ssm'] = {'repo_id': 'kernels-community/mamba-ssm', 'revision': '768416abf0ad7262a23d83c6efef10b9a4786489'}
    hk._HUB_KERNEL_MAPPING['causal-conv1d'] = {'repo_id': 'kernels-community/causal-conv1d', 'revision': '6e9a58272220a363dc50e59cba737abcedd7b59f'}


def fast_moe_forward(self, hidden_states):
    """Same arithmetic and the same order as NemotronHMoE.forward (experts in index order; within an expert, rows ordered
    by top-k position then token), but one GPU sync per block instead of several per expert. Adopted only if
    run_gate.py moecheck finds outputs identical to the original."""
    residuals, shape = hidden_states, hidden_states.shape
    topk_indices, topk_weights = self.route_tokens_to_experts(self.gate(hidden_states))
    x = self.fc1_latent_proj(hidden_states.view(-1, shape[-1]))
    out = torch.zeros_like(x, dtype=topk_weights.dtype)
    N = x.shape[0]
    flat = topk_indices.T.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=self.n_routed_experts).tolist()
    tok, pos = order % N, order // N
    xs, ws = x[tok].split(counts), topk_weights[tok, pos][:, None].split(counts)
    for e, (t, xe, we) in enumerate(zip(tok.split(counts), xs, ws)):
        if len(t):
            out.index_add_(0, t, (self.experts[e](xe) * we).to(out.dtype))
    return self.fc2_latent_proj(out.to(x.dtype)).view(*shape) + self.shared_experts(residuals)


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32')
    assert sr == 16000 and x.ndim == 1, (path, sr, x.shape)
    return x


class LRU(OrderedDict):
    def __init__(self, n):
        super().__init__(); self.n = n

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        while len(self) > self.n:
            self.popitem(last=False)
        return v


class OmniBase:
    name = None

    def _finish_init(self, layers, head):
        self.layers = layers
        self.n_layers = len(layers)
        self._w32 = head.weight.detach().float()
        self.letter_ids = {}
        for L in 'ABCDE':
            ids = self.tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, (L, ids)
            self.letter_ids[L] = ids[0]
        self._capture, self._idx, self._ablate = None, None, {}
        for l, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(l))

    # ---- hooks (as midterm) ------------------------------------------------------------------
    def _make_hook(self, l):
        def hook(module, args, output):
            if l not in self._ablate and self._capture is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if l in self._ablate:
                u, alpha = self._ablate[l]
                u = u.to(h.device)
                hf = h.float()
                h = (hf - alpha * (hf @ u).unsqueeze(-1) * u).to(h.dtype)
            if self._capture is not None:
                rows = torch.arange(h.shape[0], device=h.device)
                self._capture.append(h[rows, self._idx.to(h.device)].detach().float().cpu())
            if l in self._ablate:
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def set_ablation(self, directions=None, alpha=1.0, layers=()):
        self._ablate = {}
        if directions is None:
            return
        for l in layers:
            u = torch.as_tensor(np.asarray(directions[l]), dtype=torch.float32)
            self._ablate[l] = (u / u.norm(), float(alpha))

    # ---- scoring -----------------------------------------------------------------------------
    def logits32(self, hid):
        return hid.float().to(self._w32.device) @ self._w32.T

    def _letters(self, logits, n_opts, n_tokens):
        letters = 'ABCDE'[:n_opts]
        ids = torch.tensor([self.letter_ids[L] for L in letters], device=logits.device)
        full = torch.log_softmax(logits, -1)
        sel = logits[ids]
        top = int(logits.argmax())
        return {'letter_logp': torch.log_softmax(sel, -1).tolist(), 'argmax': letters[int(sel.argmax())],
                'letter_mass': float(full[ids].exp().sum()), 'finite': bool(torch.isfinite(logits).all()),
                'n_tokens': int(n_tokens), 'top_token': self.tok.decode([top])}

    @torch.no_grad()
    def score_many(self, prompts, n_opts, system=None, capture=False, fast=False):
        """One prompt at a time by default: the batched path failed G1(b) (OVERNIGHT_LOG.md). -> list of result dicts, list of residuals [n_layers, H] (or None)."""
        outs, resid = [], []
        toks = [self.n_tokens(p, system) for p in prompts] if fast else [0] * len(prompts)
        groups, cur = [], []
        for i in range(len(prompts)):
            if cur and (not fast or max(toks[j] for j in cur + [i]) * (len(cur) + 1) > MAX_BATCH_TOKENS):
                groups.append(cur); cur = []
            cur.append(i)
        groups.append(cur)
        for g in groups:
            self._capture = [] if capture else None
            hid, lens = self.last_hidden([prompts[i] for i in g], system)
            if capture:
                cap = torch.stack(self._capture, 1).numpy()  # [B, n_layers, H]
                assert cap.shape[1] == self.n_layers
                resid += [cap[b] for b in range(len(g))]
            self._capture = None
            for b in range(len(g)):
                outs.append(self._letters(self.logits32(hid[b]), n_opts, lens[b]))
        return outs, (resid if capture else None)


# ================================================================================================
class Nemotron(OmniBase):
    name = 'nemotron'

    def __init__(self):
        from transformers import AutoProcessor, AutoModel, AutoConfig, AutoTokenizer, AutoImageProcessor
        path = snapshot('nemotron')
        pin_kernels()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        # shipped default is flash_attention_2; the wrapper has no attention of its own, the LM uses SDPA
        cfg._attn_implementation = 'eager'; cfg.llm_config._attn_implementation = 'sdpa'; cfg.vision_config.use_flash_attn = False
        n = torch.cuda.device_count()
        self.model, self.load_info = AutoModel.from_pretrained(
            path, config=cfg, trust_remote_code=True, dtype=torch.bfloat16, device_map=self._device_map(cfg, n),
            output_loading_info=True)
        self.model.eval()
        self.lm = self.model.language_model
        # SDPA is required: without a mask it applies causal attention itself, the eager path would not
        assert self.lm.config._attn_implementation == 'sdpa', self.lm.config._attn_implementation
        self.fix_buffers()
        self.block_types = list(self.lm.config.layers_block_type)
        self.img_cache, self.aud_cache = LRU(6), LRU(2000)
        self._finish_init(self.lm.backbone.layers, self.lm.lm_head)
        self.check_moe_bias()
        mod = __import__(type(self.lm).__module__, fromlist=['x'])
        assert mod.is_fast_path_available, 'Mamba CUDA kernels not loaded'
        self.moe_cls, self.moe_orig = mod.NemotronHMoE, mod.NemotronHMoE.forward
        chk = Path(__file__).resolve().parent.parent / 'results/gate/moecheck3.json'
        self.fast_moe, self.fast_calls = False, 0
        if chk.exists() and __import__('json').loads(chk.read_text())['ok']:
            self.enable_fast()

    def enable_fast(self):
        """Remove accelerate's per-module device wrappers BELOW block level (the block-level wrapper that moves inputs
        between GPUs stays) and use the one-sync MoE dispatch. Note: those wrappers are set on module instances, so a
        class-level patch alone is bypassed; that made the first moecheck vacuous."""
        from accelerate.hooks import remove_hook_from_submodules
        for block in self.layers:
            for child in (block.norm, block.mixer):
                remove_hook_from_submodules(child)
        me = self
        def fwd(mod, h):
            me.fast_calls += 1
            return fast_moe_forward(mod, h)
        import functools
        for block, t in zip(self.layers, self.block_types):
            if t == 'moe':
                block.mixer.forward = functools.partial(fwd, block.mixer)   # instance level: accelerate leaves an instance attribute behind
        self.fast_moe = True

    def disable_fast(self):
        """Back to the original MoE dispatch (used if the fast path's gradients do not match, results/gate/dpo_check.json)."""
        import functools
        for block, t in zip(self.layers, self.block_types):
            if t == 'moe':
                block.mixer.forward = functools.partial(self.moe_orig, block.mixer)
        self.fast_moe = False

    def fix_buffers(self):
        """Two buffers are not in the checkpoint; set them to the values their constructors define."""
        rm = self.model.vision_model.radio_model
        rm.summary_idxs = torch.tensor([0, 1, 2], dtype=torch.int64, device='cuda:0')   # teachers with use_summary (unused by .features)
        ep = self.model.sound_encoder.encoder.encode_positions
        H = self.model.sound_encoder.encoder.config.hidden_size
        want = 1.0 / (10000.0 ** (torch.arange(0, H, 2, dtype=torch.float32) / H))
        self.inv_freq_was_ok = bool(torch.allclose(ep.inv_freq.float().cpu(), want, rtol=1e-2))
        ep.inv_freq = want.to('cuda:0')
        bad = [n for n, t in list(self.model.named_parameters()) + list(self.model.named_buffers()) if t.device.type == 'meta']
        assert not bad, bad[:5]

    def check_moe_bias(self):
        b = [m.e_score_correction_bias for m in self.lm.modules() if hasattr(m, 'e_score_correction_bias')]
        self.load_info_moe = {'n': len(b), 'dtype': str(b[0].dtype), 'nonzero': bool(any(x.abs().sum() > 0 for x in b))}

    @staticmethod
    def _device_map(cfg, n):
        top = {k: 0 for k in ('vision_model', 'mlp1', 'sound_encoder', 'sound_projection')}
        pat = cfg.llm_config.layers_block_type
        w = [dict(moe=2.45, mamba=0.2, attention=0.3).get(t, 0.3) for t in pat]   # approximate GB per block
        total, cum, dm = sum(w), 0.0, {}
        first = total / n + (0.7 if n > 1 else 0)   # measured: gives about 33 GB on each A100
        for i, x in enumerate(w):
            cum += x
            d = 0 if cum <= first else min(n - 1, 1 + int((cum - first) / ((total - first) / max(n - 1, 1)))) if n > 1 else 0
            dm[f'language_model.backbone.layers.{i}'] = d
        dm.update({'language_model.backbone.embeddings': 0, 'language_model.backbone.norm_f': n - 1, 'language_model.lm_head': n - 1})
        dm.update(top)
        return dm

    # ---- inputs ------------------------------------------------------------------------------
    def chat(self, parts, system=None):
        s = ''
        for kind, v in parts:
            s += {'image': '<image>\n', 'audio': '<so_embedding>'}.get(kind) or v
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': s}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                            chat_template=getattr(self.processor, 'chat_template', None) or self.tok.chat_template)

    def _img_grid(self, img):
        wp, hp = self.processor.image_processor._compute_target_patches(img, min(13312, (16384 - 4) * 4))
        return wp, hp

    # With thinking off the model opens its answer with a newline ("\nB") on 19 of 20 items (G1 attempt 1), so the
    # newline is part of the fixed assistant prefix and the scored position is the letter itself.
    ANSWER_PREFIX = '\n'

    def encode(self, parts, system=None, prefix=True):
        s = self.chat(parts, system) + (self.ANSWER_PREFIX if prefix else '')
        for kind, v in parts:
            if kind == 'image':
                wp, hp = self._img_grid(v)
                s = s.replace('<image>', '<img>' + '<|ph|>' * (wp * hp // 4) + '</img>', 1)
            elif kind == 'audio':
                n = max(1, self.processor._estimate_audio_num_embeddings(len(load_wav(v))))
                s = s.replace('<so_embedding>', '<so_start>' + '<|aph|>' * n + '<so_end>', 1)
        s = s.replace('<|ph|>', '<image>').replace('<|aph|>', '<so_embedding>')
        return self.tok(s, add_special_tokens=False)['input_ids']

    def n_tokens(self, parts, system=None):
        return len(self.encode(parts, system))

    @torch.no_grad()
    def _img_embed(self, img):
        def fn():
            pv = self.processor.image_processor(images=[img], return_tensors='pt')['pixel_values']
            pv = pv.to('cuda:0', torch.bfloat16)
            return self.model.extract_feature(pv).reshape(-1, self.lm.config.hidden_size)
        return self.img_cache.get_or(id(img), lambda: (img, fn()))[1]   # keep img alive so id() stays unique

    @torch.no_grad()
    def _aud_embed(self, path):
        def fn():
            fe = self.model.sound_feature_extractor
            a = fe([load_wav(path)], sampling_rate=fe.sampling_rate, return_tensors='pt')
            m = a.get('attention_mask', None)
            x = self.model.extract_sound_feature(a.input_features.to('cuda:0', torch.bfloat16), m.to('cuda:0') if m is not None else None)
            return x.reshape(-1, self.lm.config.hidden_size)
        return self.aud_cache.get_or(str(path), fn)

    @torch.no_grad()
    def embed(self, parts, system=None, prefix=True):
        ids = torch.tensor(self.encode(parts, system, prefix), device='cuda:0')
        e = self.lm.get_input_embeddings()(ids).clone()
        for tid, kind, fn in ((self.model.img_context_token_id, 'image', self._img_embed), (self.model.sound_context_token_id, 'audio', self._aud_embed)):
            vals = [fn(v) for k, v in parts if k == kind]
            mask = ids == tid
            if vals:
                cat = torch.cat(vals, 0).to(e)
                assert int(mask.sum()) == len(cat), (kind, int(mask.sum()), len(cat))
                e[mask] = cat
            else:
                assert not mask.any()
        return e

    def last_hidden(self, prompts, system=None):
        """Final-norm hidden state at each prompt's last token (works with or without grad)."""
        es = [self.embed(p, system) for p in prompts]
        lens = [len(e) for e in es]
        x = torch.zeros(len(es), max(lens), es[0].shape[-1], dtype=es[0].dtype, device=es[0].device)
        for b, e in enumerate(es):
            x[b, :lens[b]] = e
        self._idx = torch.tensor([l - 1 for l in lens])
        hid = self.backbone()(inputs_embeds=x, attention_mask=None, use_cache=False).last_hidden_state
        return hid[torch.arange(len(es), device=hid.device), self._idx.to(hid.device)], lens

    def backbone(self):
        return self.lm.backbone

    @torch.no_grad()
    def generate(self, parts, system=None, max_new_tokens=4):
        """Greedy, recomputing the full sequence per step (no Mamba cache): simple and exact. Free generation: no forced answer prefix."""
        e = self.embed(parts, system, prefix=False)
        out = []
        for _ in range(max_new_tokens):
            hid = self.backbone()(inputs_embeds=e[None], attention_mask=None, use_cache=False).last_hidden_state[0, -1]
            t = int(self.logits32(hid).argmax())
            out.append(t)
            if t in (11, self.tok.eos_token_id):
                break
            e = torch.cat([e, self.lm.get_input_embeddings()(torch.tensor([t], device=e.device)).to(e)], 0)
        return self.tok.decode(out, skip_special_tokens=False)

    LORA_REGEX = r'.*backbone\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj)'

    def add_lora(self, lcfg):
        from peft import get_peft_model
        self.peft = get_peft_model(self.model.language_model, lcfg)
        return self.peft


# ================================================================================================
class Qwen3Omni(OmniBase):
    name = 'qwen'

    def __init__(self):
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
        path = snapshot('qwen')
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(path)
        self.tok = self.processor.tokenizer
        self.model, self.load_info = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa', output_loading_info=True)
        self.model.eval()
        self.lm = self.model.model
        self.block_types = ['attention+moe'] * len(self.lm.layers)
        self._last = None
        self._finish_init(self.lm.layers, self.model.lm_head)
        self.model.lm_head.register_forward_pre_hook(self._head_pre)

    def _head_pre(self, module, args):
        h = args[0]
        if self._last_idx is not None:
            self._last = h[torch.arange(h.shape[0], device=h.device), self._last_idx.to(h.device)]
            return (h[:, -1:],)

    _last_idx = None

    def chat(self, parts, system=None):
        content = [{'type': k, 'text': v} if k == 'text' else {'type': k} for k, v in parts]
        msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': system}]}] if system else []) + [{'role': 'user', 'content': content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _enc(self, prompts, system=None):
        imgs = [v for p in prompts for k, v in p if k == 'image']
        auds = [load_wav(v) for p in prompts for k, v in p if k == 'audio']
        enc = self.processor(text=[self.chat(p, system) for p in prompts], images=imgs or None, audio=auds or None,
                             padding=True, padding_side='right', return_tensors='pt')
        return enc.to('cuda:0')

    def n_tokens(self, parts, system=None):
        n = sum(len(self.tok(v)['input_ids']) for k, v in parts if k == 'text') + 40
        for k, v in parts:
            if k == 'image':
                n += (v.width // 32 + 1) * (v.height // 32 + 1)
            elif k == 'audio':
                n += 150
        return n

    def last_hidden(self, prompts, system=None):
        enc = self._enc(prompts, system)
        lens = enc['attention_mask'].sum(1).tolist()
        self._idx = self._last_idx = torch.tensor([l - 1 for l in lens])
        if 'pixel_values' in enc:
            enc['pixel_values'] = enc['pixel_values'].to(torch.bfloat16)
        if 'input_features' in enc:
            enc['input_features'] = enc['input_features'].to(torch.bfloat16)
        self.model(**enc, use_cache=False)
        h, self._last, self._last_idx = self._last, None, None
        return h, lens

    @torch.no_grad()
    def generate(self, parts, system=None, max_new_tokens=4):
        enc = self._enc([parts], system)
        for k in ('pixel_values', 'input_features'):
            if k in enc:
                enc[k] = enc[k].to(torch.bfloat16)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=False)

    LORA_REGEX = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'

    def add_lora(self, lcfg):
        from peft import get_peft_model
        self.peft = get_peft_model(self.model, lcfg)
        return self.peft


# ================================================================================================
class Gemma4(OmniBase):
    """Gemma 4 12B (text, image, audio in; Nov 25 speech-transfer plan). Thinking off (the template's empty thought
    block). Scores the next token after the template's generation prompt. The output head is followed by a final logit
    softcap (30); logits32 applies it, so letter probabilities are the model's own. One prompt at a time.
    Image budget: IMG_TOKENS soft tokens for the five-frame strip (processor default 280, used here; maximum 1120)."""
    name = 'gemma'
    IMG_TOKENS = int(os.environ.get('GEMMA_IMG_TOKENS', 280))   # the processor's own default; 1120 made a pass about 1,160 tokens (10+ GPU hours in total)

    def __init__(self, device=None, dtype=torch.bfloat16):
        from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration
        path = snapshot('gemma')
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.image_processor.max_soft_tokens = self.IMG_TOKENS
        self.tok = self.processor.tokenizer
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.load_info = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            path, dtype=dtype, device_map=self.device, attn_implementation='sdpa', output_loading_info=True)
        self.model.eval()
        self.lm = self.model.model.language_model
        self.softcap = self.model.config.get_text_config().final_logit_softcapping
        self.block_types = ['attention+mlp'] * len(self.lm.layers)
        self._last = None
        self._finish_init(self.lm.layers, self.model.lm_head)
        self.model.lm_head.register_forward_pre_hook(self._head_pre)

    _last_idx = None

    def _head_pre(self, module, args):
        h = args[0]
        if self._last_idx is not None:
            self._last = h[torch.arange(h.shape[0], device=h.device), self._last_idx.to(h.device)]
            return (h[:, -1:],)

    def logits32(self, hid):
        z = hid.float().to(self._w32.device) @ self._w32.T
        return torch.tanh(z / self.softcap) * self.softcap if self.softcap else z

    def chat(self, parts, system=None):
        content = [{'type': 'text', 'text': v} if k == 'text' else {'type': k} for k, v in parts]
        msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': system}]}] if system else []) + [{'role': 'user', 'content': content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _enc(self, prompts, system=None):
        imgs = [v for p in prompts for k, v in p if k == 'image']
        auds = [load_wav(v) for p in prompts for k, v in p if k == 'audio']
        kw = {'images': imgs} if imgs else {}
        if auds:
            kw['audio'] = auds
        enc = self.processor(text=[self.chat(p, system) for p in prompts], padding=True, padding_side='right', return_tensors='pt', **kw)
        enc = enc.to(self.device)
        for k in ('pixel_values', 'input_features', 'audio_values'):
            if k in enc and enc[k].is_floating_point():
                enc[k] = enc[k].to(self.model.dtype)
        return enc

    def n_tokens(self, parts, system=None):
        return int(self._enc([parts], system)['input_ids'].shape[1])

    def last_hidden(self, prompts, system=None):
        enc = self._enc(prompts, system)
        lens = enc['attention_mask'].sum(1).tolist()
        self._idx = self._last_idx = torch.tensor([l - 1 for l in lens])
        self.model(**enc, use_cache=False)
        h, self._last, self._last_idx = self._last, None, None
        return h, lens

    @torch.no_grad()
    def generate(self, parts, system=None, max_new_tokens=4):
        enc = self._enc([parts], system)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=False)

    LORA_REGEX = r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'

    def add_lora(self, lcfg):
        """LoRA on the language model's attention projections; injected in place, so self.model runs with the adapter."""
        from peft import get_peft_model
        self.peft = get_peft_model(self.model, lcfg)
        return self.peft


def load(name):
    return {'nemotron': Nemotron, 'qwen': Qwen3Omni, 'gemma': Gemma4}[name]()
