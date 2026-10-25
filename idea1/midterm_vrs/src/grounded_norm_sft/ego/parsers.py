"""Strict terminal-answer extraction (plan A5.1). Returns zero-based indices.

Best/justification: the final nonempty line must be whitespace + one digit 1-5 + whitespace.
Sensible set: a terminal standalone bracketed block, ast.literal_eval, `type(v) is list`, every
member `type(m) is int` and 1<=m<=5; deduplicated into a set with duplicates recorded.
Never scans rationale numbers, converts letters, retries, or defaults to None.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_ACTION_LINE = re.compile(r"[ \t]*([1-5])[ \t]*")
_SENSIBLE_BLOCK = re.compile(r"(?:^|\n)[ \t]*(\[[^\[\]]*\])[ \t\r\n]*\Z")
_LETTER_LINE = re.compile(r"[ \t]*\(?([A-Fa-f])[\).:]?[ \t]*")


@dataclass(frozen=True)
class ActionParse:
    valid: bool
    index: int | None
    reason: str | None


@dataclass(frozen=True)
class SetParse:
    valid: bool
    indices: frozenset | None
    duplicates: int
    reason: str | None


def _last_nonempty_line(raw: str) -> str | None:
    lines = [x for x in raw.splitlines() if x.strip()]
    return lines[-1] if lines else None


def parse_action(raw: str) -> ActionParse:
    line = _last_nonempty_line(raw)
    if line is None:
        return ActionParse(False, None, "empty_output")
    m = _ACTION_LINE.fullmatch(line)
    if not m:
        return ActionParse(False, None, "final_line_not_single_index")
    return ActionParse(True, int(m.group(1)) - 1, None)


def parse_sensible(raw: str) -> SetParse:
    m = _SENSIBLE_BLOCK.search(raw)
    if not m:
        return SetParse(False, None, 0, "no_terminal_list")
    try:
        value = ast.literal_eval(m.group(1))
    except (SyntaxError, ValueError):
        return SetParse(False, None, 0, "literal_eval_failed")
    if type(value) is not list:
        return SetParse(False, None, 0, "not_a_list")
    if any(type(v) is not int or not 1 <= v <= 5 for v in value):
        return SetParse(False, None, 0, "member_not_int_1_to_5")
    return SetParse(True, frozenset(v - 1 for v in value), len(value) - len(set(value)), None)


def unsupported_letter_emission(raw: str) -> bool:
    """Final nonempty line is a bare A-F style letter answer (recorded, never converted)."""
    line = _last_nonempty_line(raw)
    return bool(line is not None and _LETTER_LINE.fullmatch(line))
