import ast
import json
import random
import re
from .common import ROOT, stable_hash

TEMPLATES = json.loads((ROOT / "configs/ego_prompts.json").read_text())

def ego_prompt(row, task, action=None, blind=False):
    prefix = TEMPLATES["prefix_blind" if blind else "prefix_video"]
    if task == "justification":
        behavior = row["behaviors"][action] if action is not None and 0 <= action < 4 else "None"
        return prefix + TEMPLATES[task].format(behavior=behavior, justifications=row["justifications"])
    return TEMPLATES[task].format(prefix=prefix, behaviors=row["behaviors"])

def r3_prompt(row, permutation=None):
    order = list(range(5)) if permutation is None else permutation
    choices = "\n".join(f"{chr(65+i)}. {row['options'][j]}" for i, j in enumerate(order))
    return ("Answer this question about the video. Briefly explain the evidence and causal or mental-state relationships relevant to the answer. "
            "Finish with a separate line containing Answer: followed by exactly one letter A, B, C, D, or E.\n\n"
            f"Question: {row['question']}\n{choices}")

def parse_r3(text):
    match = re.search(r"(?:^|\n)\s*Answer:\s*([A-E])\s*[.!]?\s*$", text, re.I)
    return ord(match[1].upper()) - 65 if match else -1

def parse_choice(text, action=False):
    """Match upstream last-integer parsing; retain malformed flags separately."""
    ints = re.findall(r"\d+", text)
    if not ints:
        return -1, False
    index = int(ints[-1]) - 1
    valid = index in range(5)
    if not valid:
        index = 4 if action else -1
    # Strict final-line validity prevents a reasoning number masquerading as an answer.
    strict = bool(re.search(r"(?:^|\n)\s*[1-5]\s*$", text))
    return index, valid and strict

def parse_sensible(text):
    try:
        matches = re.findall(r"\[[^\[\]]*\]", text)
        values = ast.literal_eval(matches[-1])
        if not isinstance(values, list) or any(type(x) is not int or x not in range(1, 6) for x in values):
            return [], False
        return sorted(set(x - 1 for x in values)), True
    except (IndexError, ValueError, SyntaxError):
        return [], False

def mismatch_map(rows):
    result = {}
    for row in rows:
        candidates = [r for r in rows if r["group"] != row["group"]]
        if not candidates:
            raise ValueError("Mismatched-video control requires at least two source groups")
        other = min(candidates, key=lambda r: stable_hash(row["id"] + ":" + r["id"]))
        result[row["id"]] = other
    return result

def item_metrics(row, prediction):
    if row["dataset"] == "r3":
        return {"accuracy": float(prediction.get("answer", -1) == row["answer"])}
    a = prediction.get("action", -1) == row["answer"]
    j = prediction.get("justification", -1) == row["answer"]
    predicted = set(prediction.get("sensible", []))
    target = set(row["sensible"])
    union = predicted | target
    return {"action": float(a), "justification": float(j), "joint": float(a and j),
            "sensibility_iou": len(predicted & target) / len(union) if union else 1.0}
