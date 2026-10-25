"""Exact prompt/processor boundary (plan 5.1, 6.2, A3).

Processor interface used (satisfied by the HF Qwen3.5 processor; tests use a fake):
  processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
  processor(text=[str], images=[img]|None, return_tensors="pt") -> mapping with input_ids
      (and image_grid_thw when an image is passed)
  processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=bool) -> mapping
  processor.tokenizer.convert_tokens_to_ids(str), processor.tokenizer.decode(list[int])
  processor.image_token (str), processor.image_processor.merge_size (int)

Every boundary assumption is asserted; failures raise (fail closed) rather than approximating.
Offsets are collected on a separate tokenizer call over the explicitly expanded text, and the
resulting IDs must equal the ordinary processor IDs, proving offsets do not alter inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .constants import (MAX_PROMPT_TOKENS, NORMBANK_LETTERS, NORMLENS_LETTERS, VISUAL_TOKENS_MAX,
                        VISUAL_TOKENS_MIN)
from .hashing import compact_json_bytes, sha256_bytes
from .prompts import (SELECTOR_GENERIC_USER_TEXT, SYSTEM_PROMPT, render_normbank_user,
                      render_normlens_user)


ACTION_PREFIX = "Proposed action: "  # both NormLens templates start with this exact text


class SerializationError(RuntimeError):
    pass


class MultiTokenAnswerError(SerializationError):
    """A response letter is not exactly one next token: stop for a documented multi-token CE redesign."""


class QuarantineError(SerializationError):
    """Example-level issue (overlength, visual-token budget) -> quarantine/report, never truncate."""


@dataclass(frozen=True)
class TemplateMarkers:
    """Qwen chat-template markers; verified against the pinned template at runtime."""
    assistant_start: str = "<|im_start|>assistant"
    think_open: str = "<think>"
    think_close: str = "</think>"
    vision_start: str = "<|vision_start|>"
    vision_end: str = "<|vision_end|>"


DEFAULT_MARKERS = TemplateMarkers()


def to_list(x: Any) -> list:
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    return list(x)


@dataclass
class Serialized:
    task: str
    example_id: str
    rendered: str
    expanded: str
    input_ids: list[int]
    model_inputs: Any
    final_prompt_row: int
    answer_token_ids: list[int]
    image_positions: list[int] = field(default_factory=list)
    num_visual_tokens: int = 0
    action_token_positions: list[int] = field(default_factory=list)
    completed_action_row: int | None = None
    decoded_action_span: str | None = None

    @property
    def suffix_rows(self) -> list[int]:
        """R: last token overlapping the complete ACTION span through the final prompt token."""
        if self.completed_action_row is None:
            return []
        return list(range(self.completed_action_row, self.final_prompt_row + 1))

    @property
    def prompt_sha256(self) -> str:
        return sha256_bytes(compact_json_bytes(self.input_ids))

    def metadata(self) -> dict:
        return {
            "task": self.task, "example_id": self.example_id, "prompt_sha256": self.prompt_sha256,
            "num_tokens": len(self.input_ids), "num_visual_tokens": self.num_visual_tokens,
            "image_first": self.image_positions[0] if self.image_positions else None,
            "image_last": self.image_positions[-1] if self.image_positions else None,
            "completed_action_row": self.completed_action_row, "final_prompt_row": self.final_prompt_row,
            "answer_token_ids": self.answer_token_ids, "decoded_action_span": self.decoded_action_span,
        }


def build_messages(user_text: str, image: Any | None, system: str | None = SYSTEM_PROMPT) -> list[dict]:
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})  # image first (A3)
    content.append({"type": "text", "text": user_text})
    msgs: list[dict] = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    return msgs


def render_prompt(processor, messages: list[dict]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def check_nonthinking_prefix(rendered: str, markers: TemplateMarkers = DEFAULT_MARKERS) -> str:
    """Require the native closed (empty) thinking block right after the final assistant start."""
    idx = rendered.rfind(markers.assistant_start)
    if idx < 0:
        raise SerializationError("generation prompt lacks assistant start marker")
    tail = rendered[idx + len(markers.assistant_start):]
    pattern = r"\s*" + re.escape(markers.think_open) + r"\s*" + re.escape(markers.think_close) + r"\s*"
    if not re.fullmatch(pattern, tail):
        raise SerializationError(f"generation prefix is not a closed empty thinking block: {tail!r}")
    return tail


def check_system_absent(rendered: str, system_marker: str = "<|im_start|>system") -> None:
    """Ego prompts are user-only; the pinned template must not inject a default system turn."""
    if system_marker in rendered:
        raise SerializationError("template injected a system message into a user-only conversation")


def expand_image_placeholders(rendered: str, image_token: str, counts: Sequence[int]) -> str:
    if rendered.count(image_token) != len(counts):
        raise SerializationError(
            f"expected {len(counts)} image placeholder(s), found {rendered.count(image_token)}")
    pieces = rendered.split(image_token)
    out = pieces[0]
    for c, nxt in zip(counts, pieces[1:]):
        out += image_token * c + nxt
    return out


def locate_unique(haystack: str, needle: str) -> int:
    first = haystack.find(needle)
    if first < 0 or haystack.find(needle, first + 1) >= 0:
        raise SerializationError("text segment not found exactly once in serialized prompt")
    return first


def overlapping_tokens(offsets: Sequence[Sequence[int]], start: int, end: int) -> list[int]:
    """Token indices whose nonempty character span overlaps [start, end)."""
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]


def visual_token_counts(model_inputs, merge_size: int) -> list[int]:
    grid = model_inputs.get("image_grid_thw") if hasattr(model_inputs, "get") else None
    if grid is None:
        return []
    out = []
    for row in (grid.tolist() if hasattr(grid, "tolist") else grid):
        t, h, w = (int(v) for v in row)
        prod = t * h * w
        if prod % (merge_size ** 2):
            raise SerializationError("grid not divisible by merge_size^2")
        out.append(prod // (merge_size ** 2))
    return out


def answer_token_ids(tokenizer, expanded_prompt: str, letters: Sequence[str]) -> list[int]:
    """Assert prompt tokens are an exact prefix and each letter is exactly one next token."""
    base = to_list(tokenizer(expanded_prompt, add_special_tokens=False)["input_ids"])
    ids = []
    for letter in letters:
        full = to_list(tokenizer(expanded_prompt + letter, add_special_tokens=False)["input_ids"])
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise MultiTokenAnswerError(f"answer {letter!r} is not a single next token at the boundary")
        if tokenizer.decode([full[-1]]) != letter:
            raise MultiTokenAnswerError(f"answer token for {letter!r} decodes differently")
        ids.append(int(full[-1]))
    if len(set(ids)) != len(ids):
        raise MultiTokenAnswerError("answer letters share token IDs")
    return ids


class Serializer:
    """Builds native inputs for NormLens / NormBank / selector-query / Ego prompts."""

    def __init__(self, processor, markers: TemplateMarkers = DEFAULT_MARKERS,
                 visual_range: tuple[int, int] = (VISUAL_TOKENS_MIN, VISUAL_TOKENS_MAX),
                 max_tokens: int = MAX_PROMPT_TOKENS):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.markers = markers
        self.visual_range = visual_range
        self.max_tokens = max_tokens
        self.image_token = processor.image_token
        self.merge_size = int(processor.image_processor.merge_size)
        self.image_token_id = self._single_token_id(self.image_token)
        self.vision_start_id = self._single_token_id(markers.vision_start)
        self.vision_end_id = self._single_token_id(markers.vision_end)

    def _single_token_id(self, token: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        enc = to_list(self.tokenizer(token, add_special_tokens=False)["input_ids"])
        if tid is None or enc != [tid]:
            raise SerializationError(f"marker {token!r} is not a single registered token")
        return int(tid)

    # ------------------------------------------------------------------ core
    def _encode(self, task: str, example_id: str, messages: list[dict], image: Any | None,
                letters: Sequence[str] | None, focus_text: str | None, user_text: str,
                require_system_absent: bool = False, focus_offset: int = 0) -> Serialized:
        rendered = render_prompt(self.processor, messages)
        check_nonthinking_prefix(rendered, self.markers)
        if require_system_absent:
            check_system_absent(rendered)
        model_inputs = self.processor(text=[rendered], images=[image] if image is not None else None,
                                      return_tensors="pt")
        ids = [int(x) for x in to_list(model_inputs["input_ids"])]
        counts = visual_token_counts(model_inputs, self.merge_size) if image is not None else []
        if image is not None and len(counts) != 1:
            raise SerializationError("exactly one image expected")
        expanded = expand_image_placeholders(rendered, self.image_token, counts) if counts else rendered
        tok = self.tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
        if [int(x) for x in to_list(tok["input_ids"])] != ids:
            raise SerializationError("offset tokenization differs from processor input_ids")
        offsets = [tuple(int(v) for v in o) for o in tok["offset_mapping"]]
        if len(ids) > self.max_tokens:
            raise QuarantineError(f"prompt has {len(ids)} tokens > {self.max_tokens}; never truncated")

        image_positions = [i for i, t in enumerate(ids) if t == self.image_token_id]
        if image is not None:
            n = counts[0]
            lo, hi = self.visual_range
            if not lo <= n <= hi:
                raise QuarantineError(f"{n} merged visual tokens outside [{lo},{hi}]")
            if len(image_positions) != n or image_positions != list(range(image_positions[0], image_positions[0] + n)):
                raise SerializationError("image placeholder tokens are not one contiguous run of the grid size")
            if ids[image_positions[0] - 1] != self.vision_start_id or ids[image_positions[-1] + 1] != self.vision_end_id:
                raise SerializationError("image run is not bracketed by vision start/end tokens")
        elif image_positions:
            raise SerializationError("image tokens present in a text-only prompt")

        ser = Serialized(task=task, example_id=example_id, rendered=rendered, expanded=expanded, input_ids=ids,
                         model_inputs=model_inputs, final_prompt_row=len(ids) - 1,
                         answer_token_ids=answer_token_ids(self.tokenizer, expanded, letters) if letters else [],
                         image_positions=image_positions, num_visual_tokens=counts[0] if counts else 0)

        if focus_text is not None:
            u0 = locate_unique(expanded, user_text)
            f_rel = focus_offset
            if user_text[f_rel:f_rel + len(focus_text)] != focus_text:
                raise SerializationError("focus text is not at its declared offset in the user text")
            start = u0 + f_rel
            end = start + len(focus_text)
            toks = overlapping_tokens(offsets, start, end)
            if not toks:
                raise SerializationError("no tokens overlap the focus span")
            decoded = self.tokenizer.decode(ids[toks[0]:toks[-1] + 1])
            if focus_text not in decoded:
                raise SerializationError("decoded span does not contain the focus text")
            if image_positions and toks[0] <= image_positions[-1]:
                raise SerializationError("focus span does not follow the image")
            ser.action_token_positions = toks
            ser.completed_action_row = toks[-1]
            ser.decoded_action_span = decoded
        return ser

    # ------------------------------------------------------------------ tasks
    # The text-only condition is always an explicit argument; it is never inferred from image=None.
    def normlens(self, example_id: str, action: str, image: Any | None, *, text_only: bool) -> Serialized:
        _check_condition(image, text_only)
        if not isinstance(action, str) or not action.strip():
            raise QuarantineError("empty action")
        user = render_normlens_user(action, text_only=text_only)
        return self._encode("normlens_text_only" if text_only else "normlens", example_id,
                            build_messages(user, image), image, NORMLENS_LETTERS, action, user,
                            focus_offset=len(ACTION_PREFIX))

    def normbank(self, example_id: str, setting: str, behavior: str, constraints: str) -> Serialized:
        user = render_normbank_user(setting, behavior, constraints)
        return self._encode("normbank", example_id, build_messages(user, None), None, NORMBANK_LETTERS, None, user)

    def selector_generic(self, image_id: str, image: Any) -> Serialized:
        """Generic image query under matched system framing (5.2); 'action' row = last query token."""
        _check_condition(image, text_only=False)
        user = SELECTOR_GENERIC_USER_TEXT
        return self._encode("selector_generic", image_id, build_messages(user, image), image, None, user, user)

    def ego(self, task: str, item_id: str, user_text: str, image: Any | None, *, text_only: bool,
            visual_range: tuple[int, int] | None = None) -> Serialized:
        _check_condition(image, text_only)
        saved = self.visual_range
        if visual_range is not None:
            self.visual_range = visual_range
        try:
            return self._encode(task, item_id, build_messages(user_text, image, system=None), image, None, None,
                                 user_text, require_system_absent=True)
        finally:
            self.visual_range = saved


def _check_condition(image: Any | None, text_only: bool) -> None:
    if type(text_only) is not bool:
        raise SerializationError("text_only must be an explicit bool")
    if text_only and image is not None:
        raise SerializationError("text-only condition must not receive an image")
    if not text_only and image is None:
        raise SerializationError("visual condition requires an image; text-only must be requested explicitly")


# --------------------------------------------------------------------------- manifest adapters
# Prompt inputs come only from these allowed fields; labels, votes, explanations, captions,
# image_src, agreement and provenance in the manifest row are never read here.
NORMLENS_PROMPT_FIELDS = ("example_id", "action")
NORMBANK_PROMPT_FIELDS = ("example_id", "setting", "behavior", "constraints")


def serialize_normlens_example(serializer: Serializer, example: dict, image: Any | None, *,
                               text_only: bool) -> Serialized:
    return serializer.normlens(example["example_id"], example["action"], image, text_only=text_only)


def serialize_normbank_example(serializer: Serializer, example: dict) -> Serialized:
    return serializer.normbank(example["example_id"], example["setting"], example["behavior"],
                               example["constraints"])


def assert_shared_prefix(a: Serialized, b: Serialized) -> int:
    """For suffix-variation tests: IDs must be identical through the completed-action row."""
    if a.completed_action_row is None or a.completed_action_row != b.completed_action_row:
        raise SerializationError("completed-action rows differ")
    r = a.completed_action_row
    if a.input_ids[:r + 1] != b.input_ids[:r + 1]:
        raise SerializationError("token prefix through the completed action differs")
    return r
