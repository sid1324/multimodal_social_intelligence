"""Prompts, output parsing and rewards for the EgoNormia GRPO pilot (shared by the Bridges trainer and the local scorer).

Formats
  social  (R0):        <think>...</think> then {"action": k, "justification": k}
  spatial (R1, R2, R3): <spatial>{...}</spatial> then <think>...</think> then the answer JSON
Rewards
  r_norm    = 0.5*[action correct] + 0.5*[justification correct] + 0.1*[answer JSON parseable]
  r_spatial = mean correctness over n_people, nearest_person, facing, hands_visible (0 if the block is unparseable)
  R0, R2: r_norm   R1: r_norm + 0.5*r_spatial   R3: r_spatial
"""
import json
import re

CONTEXT = ("The image is a left-to-right grid of five chronological frames from an egocentric (head-mounted) camera, "
           "captured before the camera wearer's next action. Choose the single most socially appropriate next action for the "
           "camera wearer from the action options, then choose the justification that best fits the action you chose.")

SPATIAL_DEFS = (
    'Before answering, describe the scene\'s spatial layout in a <spatial> block with exactly these fields:\n'
    '- "n_people": the largest number of people visible in any one of the five frames (0, 1, 2, or 3 meaning 3 or more); '
    'body parts of the camera wearer count if a detector would see a person.\n'
    '- "nearest_person": how close the nearest visible person is to the camera wearer, relative to typical everyday '
    'egocentric scenes: "near" (among the closest third of scenes), "mid", "far" (among the farthest third), or "none" if no '
    'person is visible.\n'
    '- "facing": "toward" if a visible person\'s body faces the camera wearer in the middle frame, "away" if people are '
    'visible but turned away or sideways, "none" if no person\'s body is visible.\n'
    '- "hands_visible": true if any hands (including the camera wearer\'s own) are visible in any frame, else false.')

FORMAT = {
    'social': ('First reason step by step inside <think>...</think>. Then output only this JSON and nothing after it: '
               '{"action": <1-5>, "justification": <1-5>}'),
    'spatial': ('Output, in this order: <spatial>{"n_people": <0-3>, "nearest_person": "near|mid|far|none", '
                '"facing": "toward|away|none", "hands_visible": true|false}</spatial>, then reason step by step inside '
                '<think>...</think>, then only this JSON and nothing after it: {"action": <1-5>, "justification": <1-5>}'),
}
ARMS = {'R0': 'social', 'R1': 'spatial', 'R2': 'spatial', 'R3': 'spatial'}
ANSWER = re.compile(r'\{\s*"action"\s*:\s*"?([1-5])"?\s*,\s*"justification"\s*:\s*"?([1-5])"?\s*\}')
SPATIAL_BLOCK = re.compile(r'<spatial>\s*(\{.*?\})\s*</spatial>', re.S)


def prompt(fmt, actions, justifications):
    lines = [CONTEXT, '']
    if fmt == 'spatial':
        lines += [SPATIAL_DEFS, '']
    lines += ['Action options:'] + [f'{i}. {o}' for i, o in enumerate(actions, 1)]
    lines += ['', 'Justification options:'] + [f'{i}. {o}' for i, o in enumerate(justifications, 1)]
    lines += ['', FORMAT[fmt]]
    return '\n'.join(lines)


def parse_answer(text):
    """Last answer JSON after the reasoning; positions are 1-based, or None."""
    tail = text.split('</think>')[-1]
    found = ANSWER.findall(tail)
    if not found:
        return None, None
    a, j = found[-1]
    return int(a), int(j)


def parse_spatial(text):
    m = SPATIAL_BLOCK.search(text)
    if not m:
        return None
    try:
        block = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return block if isinstance(block, dict) else None


def spatial_score(block, label):
    """Mean correctness over the four fields against the extractor labels (n_people, nearest_dist, facing, hands)."""
    if block is None or label is None:
        return 0.0
    hits = []
    try:
        hits.append(min(int(block.get('n_people')), 3) == int(label['n_people']))
    except (TypeError, ValueError):
        hits.append(False)
    hits.append(str(block.get('nearest_person', '')).lower() == str(label['nearest_dist']))
    hits.append(str(block.get('facing', '')).lower() == str(label['facing']))
    hv = block.get('hands_visible')
    hv = hv if isinstance(hv, bool) else str(hv).lower() == 'true' if str(hv).lower() in ('true', 'false') else None
    hits.append(hv is not None and hv == bool(int(label['hands'])))
    return sum(hits) / 4


def norm_score(text, order, gold):
    """order[kind][p-1] = original option index at displayed position p. Returns (r_norm, parsed, action_ok, just_ok)."""
    a, j = parse_answer(text)
    if a is None:
        return 0.0, False, False, False
    a_ok = order['action'][a - 1] == gold
    j_ok = order['justification'][j - 1] == gold
    return 0.5 * a_ok + 0.5 * j_ok + 0.1, True, a_ok, j_ok


def reward(arm, text, order, gold, label):
    rn, parsed, a_ok, j_ok = norm_score(text, order, gold)
    rs = spatial_score(parse_spatial(text), label) if ARMS[arm] == 'spatial' else 0.0
    total = {'R0': rn, 'R1': rn + 0.5 * rs, 'R2': rn, 'R3': rs}[arm]
    return total, {'r_norm': rn, 'r_spatial': rs, 'parsed': parsed, 'action': a_ok, 'justification': j_ok}
