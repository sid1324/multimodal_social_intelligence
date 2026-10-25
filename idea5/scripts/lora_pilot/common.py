"""Shared model loading and message building for the LoRA pilot (Qwen3-VL-8B-Instruct, bf16)."""
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
LORA_TARGETS = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'


def load(adapter=None, train=False):
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    processor = AutoProcessor.from_pretrained(MODEL)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    return model, processor


def messages(image_paths, text, answer=None):
    content = [{'type': 'image', 'image': Image.open(ROOT / p if not Path(p).is_absolute() else p).convert('RGB')} for p in image_paths]
    content.append({'type': 'text', 'text': text})
    conv = [{'role': 'user', 'content': content}]
    if answer is not None:
        conv.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]})
    return conv
