"""V100-compatible NF4 model and deterministic video preprocessing."""
import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import STORE, config


def sample_video(path, num_frames=8, max_side=384, end_seconds=None):
    """Cache only pre-sampled RGB frames and original frame timestamps."""
    import av
    path = Path(path)
    key = hashlib.sha256(f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{num_frames}:{max_side}:{end_seconds}".encode()).hexdigest()
    cache = STORE / "data/frame_cache" / (key + ".npz")
    if cache.exists():
        with np.load(cache) as d:
            return d["frames"], {"fps": float(d["fps"]), "total_num_frames": int(d["total"]), "frames_indices": d["indices"].tolist()}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        if fps <= 0:
            raise ValueError(f"No valid frame rate: {path}")
        # Count first when stream.frames is unknown; never retain full-resolution clip.
        total = stream.frames
        if not total:
            total = sum(1 for _ in container.decode(video=0))
    if total < 1:
        raise ValueError(f"No frames: {path}")
    if end_seconds is not None:
        total = min(total, max(1, int(end_seconds * fps)))
    indices = np.linspace(0, total - 1, num_frames).round().astype(int)
    wanted = set(indices.tolist())
    selected = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                im = frame.to_image()
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                selected[i] = np.asarray(im.convert("RGB"))
            if i >= indices[-1]:
                break
    if any(i not in selected for i in indices):
        raise ValueError(f"Decoded fewer frames than metadata: {path}")
    frames = np.stack([selected[i] for i in indices])
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Distinct process temp names protect shared cache when evaluating independently.
    import os
    temp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, frames=frames, fps=fps, total=total, indices=indices)
    temp.replace(cache)
    return frames, {"fps": fps, "total_num_frames": total, "frames_indices": indices.tolist()}


class VideoPolicy:
    def __init__(self, adapter=None, trainable=False):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        self.cfg = config()
        model_path = STORE / "models/Qwen3.5-9B"
        if not (model_path / "READY").exists():
            raise RuntimeError("Pinned model download is not complete (READY missing)")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=["visual"])
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, quantization_config=quant, dtype=torch.float16,
            device_map={"": 0}, attn_implementation="sdpa", local_files_only=True,
        )
        # Apply identical frozen-backbone precision preparation for baseline and
        # adapter evaluation, so before/after differences are not dtype changes.
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=trainable,
                                                     gradient_checkpointing_kwargs={"use_reentrant": False})
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=trainable)
        elif trainable:
            suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"}
            targets = [n for n, _ in self.model.named_modules()
                       if ".language_model.layers." in n and n.rsplit(".", 1)[-1] in suffixes]
            if not targets:
                raise RuntimeError("No language-only LoRA targets found")
            self.model = get_peft_model(self.model, LoraConfig(r=self.cfg["lora_rank"],
                lora_alpha=self.cfg["lora_alpha"], lora_dropout=0.0, target_modules=targets,
                bias="none", task_type="CAUSAL_LM"))
        self.model.eval()
        self.device = torch.device("cuda:0")

    def inputs(self, prompt, video_path=None, synthetic=False, end_seconds=None):
        content = []
        kwargs = {}
        if video_path or synthetic:
            if synthetic:
                frames = np.zeros((8, 192, 256, 3), dtype=np.uint8)
                frames[:, :, :, 0] = np.arange(8, dtype=np.uint8)[:, None, None] * 30
                metadata = {"fps": 2.0, "total_num_frames": 8, "frames_indices": list(range(8))}
            else:
                frames, metadata = sample_video(video_path, self.cfg["frames"], self.cfg["frame_max_side"], end_seconds)
            # Transformers 5.3.0 expands one video into a separate video-token
            # run per temporal patch, but supplies only one video_grid_thw row.
            # Qwen3.5 RoPE expects one grid row per run. Present the same eight
            # ordered frames as four two-frame clips, preserving timestamps.
            # Each clip is one temporal patch and therefore one matching grid.
            if len(frames) % 2:
                raise ValueError("Video frame count must be even for temporal patches")
            chunks = [frames[i:i + 2] for i in range(0, len(frames), 2)]
            metadata_chunks = [{**metadata, "frames_indices": metadata["frames_indices"][i:i + 2]}
                               for i in range(0, len(frames), 2)]
            content.extend({"type": "video"} for _ in chunks)
            kwargs.update(videos=chunks, video_metadata=metadata_chunks, do_sample_frames=False,
                          size={"shortest_edge": 128 * 32 * 32, "longest_edge": self.cfg["frames"] * self.cfg["frame_max_side"] ** 2})
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.cfg["enable_thinking"])
        batch = self.processor(text=[text], return_tensors="pt", **kwargs)
        return batch.to(self.device)

    def generate(self, inputs, sample=False, max_new_tokens=None):
        self.model.eval()
        opts = {"max_new_tokens": max_new_tokens or self.cfg["max_new_tokens"], "do_sample": sample,
                "use_cache": True, "pad_token_id": self.processor.tokenizer.pad_token_id}
        if sample:
            # Sampling distribution equals the unwarped model distribution used in policy gradients.
            opts.update(temperature=1.0, top_p=1.0, top_k=0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = self.model.generate(**inputs, **opts)
        tokens = output[:, inputs["input_ids"].shape[1]:].clone()
        text = self.processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return tokens, text

    def token_logps(self, inputs, completion):
        # Only materialize logits for completion positions; vocab is ~248k tokens.
        full = dict(inputs)
        full["input_ids"] = torch.cat([inputs["input_ids"], completion], dim=1)
        full["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(completion)], dim=1)
        if "mm_token_type_ids" in full:
            full["mm_token_type_ids"] = torch.cat([full["mm_token_type_ids"], torch.zeros_like(completion)], dim=1)
        length = completion.shape[1]
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(**full, use_cache=False, logits_to_keep=length + 1)
        logits = out.logits[:, :-1, :].float()
        return logits.gather(-1, completion.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    def reference_context(self):
        return self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else contextlib.nullcontext()

    def save(self, directory):
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)

    def memory(self):
        return {"allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
