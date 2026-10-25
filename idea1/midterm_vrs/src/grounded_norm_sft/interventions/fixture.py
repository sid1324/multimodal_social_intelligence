"""Minimal Qwen-like fake processor for CPU serializer tests (not the real tokenizer).

Tokens: registered special markers | words | single other characters. Offsets are exact.
Images are dicts {"grid": (t, h, w)}; merged visual tokens = t*h*w // merge_size**2.
"""

import re

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>",
            "<think>", "</think>"]
_PATTERN = re.compile("|".join(re.escape(s) for s in SPECIALS) + r"|\w+|[^\w]", re.UNICODE)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {s: i for i, s in enumerate(SPECIALS)}
        self.inv = dict(enumerate(SPECIALS))

    def _id(self, tok):
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
            self.inv[self.vocab[tok]] = tok
        return self.vocab[tok]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(_PATTERN.finditer(text))
        out = {"input_ids": [self._id(m.group(0)) for m in matches]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return out

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok)

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


class _ImageProcessor:
    merge_size = 2


class FakeProcessor:
    image_token = "<|image_pad|>"

    def __init__(self, system_default=False):
        self.tokenizer = FakeTokenizer()
        self.image_processor = _ImageProcessor()
        self.system_default = system_default

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        out = ""
        if self.system_default and messages[0]["role"] != "system":
            out += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "".join("<|vision_start|><|image_pad|><|vision_end|>" if c["type"] == "image" else c["text"]
                                  for c in content)
            out += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
        return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, images=None, return_tensors=None):
        (s,) = text
        grids = []
        if images:
            assert s.count(self.image_token) == len(images)
            pieces = s.split(self.image_token)
            s = pieces[0]
            for img, nxt in zip(images, pieces[1:]):
                t, h, w = img["grid"]
                grids.append([t, h, w])
                s += self.image_token * (t * h * w // 4) + nxt
        out = {"input_ids": [self.tokenizer(s)["input_ids"]]}
        if grids:
            out["image_grid_thw"] = grids
        return out

import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from ..serializer import Serializer

class ImageProcessor(FakeProcessor):
    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if kwargs.get('images'):
            # Exact synthetic image patch data, deterministic across generic/action.
            out['mm_token_type_ids'] = [[int(i == self.tokenizer.convert_tokens_to_ids(self.image_token)) for i in out['input_ids'][0]]]
            out['pixel_values'] = torch.arange(16 * 24).reshape(16, 24).float() / 384
        return out



def build():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    serializer = Serializer(ImageProcessor(), visual_range=(1, 256))
    image = {'grid': (1, 4, 4)}
    s = serializer.normlens('sample', 'Run', image, text_only=False)
    ids = dict(image_token_id=serializer.image_token_id, vision_start_id=serializer.vision_start_id,
               vision_end_id=serializer.vision_end_id, merge_size=2)
    tc = Qwen3_5TextConfig(vocab_size=512, hidden_size=32, intermediate_size=40,
        num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=2, head_dim=4,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(32)],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .5}, use_cache=False)
    vc = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
        num_position_embeddings=16)
    config = Qwen3_5Config(text_config=tc, vision_config=vc, image_token_id=ids['image_token_id'],
        vision_start_token_id=ids['vision_start_id'], vision_end_token_id=ids['vision_end_id'], video_token_id=511)
    config._attn_implementation = 'eager'
    model = native.Qwen3_5ForConditionalGeneration(config).requires_grad_(False).eval()
    for m in model.modules():
        if type(m) is native.Qwen3_5GatedDeltaNet:
            with torch.no_grad():
                m.A_log.fill_(-3.)
                m.dt_bias.fill_(-2.)
    return model, serializer, s, ids, image


def run(output):
    from pathlib import Path
    from .pipeline import cache_inputs, validate_index, Budget
    from .cache import digest
    from .potency import run as potency_run
    from ..io_utils import write_json
    model, serializer, item, alignment, image = build()
    pins = {k: 'a'*64 for k in ('model_sha256', 'processor_sha256', 'template_sha256')}
    class Inputs(dict):
        identity = dict(fixture=True, source='deterministic_synthetic_grid_v1')
        image_sha256 = {'sample': 'b'*64}
    inputs = Inputs(sample=item)
    path, state = cache_inputs(model, serializer, inputs, output, pins=pins,
        alignment_ids=alignment, image_for=lambda key: (image, 'fixture-image'), budget=Budget(2, .1))
    again, second = cache_inputs(model, serializer, inputs, output, pins=pins,
        alignment_ids=alignment, image_for=lambda key: (image, 'fixture-image'), budget=Budget(2, .1))
    assert path == again and state == second
    paths, _ = validate_index(path, inputs, pins=pins, alignment_ids=alignment)
    evidence = potency_run(model, inputs, {}, cache_paths=paths, pins=pins,
        image_sha256=inputs.image_sha256, alignment_ids=alignment, shuffle_map={}, shuffle_map_sha256=digest({}))
    evidence['fixture_mode'] = True
    write_json(Path(output) / 'fixture-potency.json', evidence)
    return dict(fixture_mode=True, native_cache_complete=True, cache_resume_verified=True,
        cohort_status=evidence['report']['status'], index=str(path), intervention_training_allowed=False,
        production_train_export_c2='unsupported')


from copy import deepcopy
from .reference import acquire
from .cache import build_record, write_record
from .potency import assignments
from .pipeline import Budget
from .training import CheckpointDev, train_model, REDUCTION
from .trainer import HybridTrainer
from ..runtime.lora import apply_lora, expected_counts
from ..schedules import build_training_schedule


def training_fixture(tmp_path, epochs=1, train_count=1, arm='H-T'):
    base, serializer, sample, alignment, image = build()
    pins = {k:'a'*64 for k in ('model_sha256','processor_sha256','template_sha256')}
    probabilities = acquire(base, sample, alignment_ids=alignment).probabilities
    class Inputs(dict):
        identity = dict(fixture_mode=True, source='native_random_cpu_only')
        @property
        def image_sha256(self):
            return {key:'b'*64 for key in self}
        @property
        def targets(self):
            return {key:[.2,.3,.5] for key in self}
    dev = Inputs()
    paths = {}
    for i in range(64):
        item = deepcopy(sample); item.example_id = 'dev%03d'%i
        dev[item.example_id] = item
        record = build_record(item, probabilities, pins=dict(pins,image_sha256='b'*64), alignment_ids=alignment)
        paths[item.example_id] = write_record(tmp_path/'cache', record)
    train_inputs, train_paths = {}, {}
    for i in range(train_count):
        train = deepcopy(sample)
        train.example_id = 'train%03d'%i
        train_inputs[train.example_id] = train
        record = build_record(train, probabilities, pins=dict(pins,image_sha256='b'*64), alignment_ids=alignment)
        train_paths[train.example_id] = write_record(tmp_path/'cache',record)
    nb = serializer.normbank('bank','home','Run','')
    count = expected_counts(base.config.text_config.to_dict())['total']
    model,_ = apply_lora(base,17,expected_total=count)
    schedule = build_training_schedule(list(train_inputs),['bank'],17,epochs,arm)
    trainer = HybridTrainer(model,schedule,train_inputs,{key:[.2,.3,.5] for key in train_inputs},
        {'bank':nb},{'bank':[.3,.3,.4]},cache_paths=train_paths,pins=pins,
        image_sha256={key:'b'*64 for key in train_inputs},alignment_ids=alignment,
        protocol_identity=dict(fixture_mode=True,checkpoint_reduction=REDUCTION),expected_lora_total=count)
    budget = Budget(10,.1)
    callback = CheckpointDev(model,dev,paths,assignments(dev),tmp_path/'run'/'dev',pins=pins,
        alignment_ids=alignment,seed=17,operator_sha256='c'*64,arm='H-T',budget=budget)
    return model,trainer,callback,budget



def run_training(output):
    """Random CPU-only model, two tiny updates; never a production approval."""
    from pathlib import Path
    root = Path(output)
    model, trainer, callback, budget = training_fixture(root, epochs=2)
    first = train_model(model, trainer, callback, root/'run', budget=budget,
        max_updates=1, update_reserve_s=.01, dev_reserve_s=.01)
    if first['status'] != 'paused' or first['next_update'] != 1:
        raise ValueError('fixture did not pause after first epoch checkpoint')
    model, trainer, callback, budget = training_fixture(root, epochs=2)
    second = train_model(model, trainer, callback, root/'run', budget=budget,
        resume=True, update_reserve_s=.01, dev_reserve_s=.01)
    return dict(fixture_mode=True, production_authorized=False, device='cpu',
                checkpoint_resume_verified=second['status']=='complete',
                c2_status=second['c2']['status'], native_family_counts=[
                    {k:v['count'] for k,v in c['summary']['families'].items()}
                    for c in second['c2']['checkpoints']], run_dir=second['run_dir'])
