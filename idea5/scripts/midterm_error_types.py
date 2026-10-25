"""Failure analysis: what kind of mistake is each wrong direct answer?

For every item a model answered wrong in a condition (joint), GPT-5.6 sees the RGB strip, the correct action and
justification, and the chosen ones, and assigns one cause:
  spatial     misjudged layout: distance, facing, reach, who is in whose path, left/right
  perception  misread who or what is present, or the activity under way
  norm        perceived the scene correctly but weighed the social norm or priority wrongly
  ambiguous   the chosen answer is also defensible from the frames
One isolated call per item (the strip is the only file in its directory). --judge gemini uses Gemini 4 instead,
as a second rater for agreement.
Output: output/idea5-midterm/error_types/<model>_<cond>[_<judge>].json  {id: {"cause": ..., "reason": ...}}

usage: midterm_error_types.py --model gpt --cond rgb [--judge gpt|gemini] [--workers 6]
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
GRIDS = MID / 'grids'
CAUSES = ('spatial', 'perception', 'norm', 'ambiguous')
PROMPT = ('rgb.jpg shows frames from an egocentric video. A model had to choose what the camera wearer should do next '
          'and why. Below are the correct answer and the model\'s wrong answer. Classify the main cause of the mistake '
          'as exactly one of: "spatial" (misjudged the spatial layout: distances, which way people face, what is within '
          'reach, who is in whose path, left/right), "perception" (misread who or what is present or what activity is '
          'under way), "norm" (saw the scene correctly but weighed the social norm or priority wrongly), "ambiguous" '
          '(the chosen answer is also defensible from the frames). Return only JSON: '
          '{"cause": "<one of the four>", "reason": "<one sentence>"}\n\n')


def wrong_items(model, cond):
    tasks = {t['id']: t for t in json.loads((MID / 'lora_pilot/egonormia_tasks.json').read_text())}
    key = json.loads((MID / 'lora_pilot/answer_key.json').read_text())
    out = []
    for f in sorted((MID / 'direct_eval' / model / cond).glob('*.json')):
        r, k = json.loads(f.read_text()), key[f.stem]
        if not (r['action_position'] and r['justification_position']):
            continue
        a, j = k['action_order'][r['action_position'] - 1], k['justification_order'][r['justification_position'] - 1]
        if a == k['gold'] and j == k['gold']:
            continue
        t = tasks[f.stem]
        ga, gj = k['action_order'].index(k['gold']), k['justification_order'].index(k['gold'])
        out.append((f.stem, f"Correct action: {t['action_options'][ga]}\nCorrect justification: {t['justification_options'][gj]}\n"
                            f"Chosen action: {t['action_options'][r['action_position'] - 1]}\n"
                            f"Chosen justification: {t['justification_options'][r['justification_position'] - 1]}"))
    return out


def classify(item, judge='gpt'):
    item_id, text = item
    for _ in range(3):
        with tempfile.TemporaryDirectory(prefix='err_') as tmp:
            shutil.copy(GRIDS / f'{item_id}_rgb.jpg', Path(tmp) / 'rgb.jpg')
            out = Path(tmp) / 'reply.txt'
            if judge == 'gemini':
                proc = subprocess.run(['agy', '-p', PROMPT + text, '--model', 'gemini-4-argon-medium', '--output-format',
                                       'json'], cwd=tmp, capture_output=True, text=True, timeout=600)
                start = proc.stdout.find('{"conversation_id"')
                out.write_text(json.loads(proc.stdout[start:].splitlines()[0])['response'] if start >= 0 else '')
            else:
                subprocess.run(['codex', 'exec', '-m', 'gpt-5.6-sol', '--sandbox', 'read-only', '--skip-git-repo-check',
                                '--ephemeral', '-C', tmp, '-o', str(out), '-i', str(Path(tmp) / 'rgb.jpg'), '-'],
                               input=PROMPT + text, capture_output=True, text=True, timeout=600)
            reply = out.read_text() if out.exists() else ''
        try:
            body = json.loads(reply[reply.find('{'):reply.rfind('}') + 1])
            if body.get('cause') in CAUSES:
                return item_id, body
        except ValueError:
            continue
    return item_id, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cond', required=True)
    ap.add_argument('--judge', default='gpt', choices=['gpt', 'gemini'])
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    dest = MID / 'error_types' / f"{args.model}_{args.cond}{'' if args.judge == 'gpt' else '_' + args.judge}.json"
    dest.parent.mkdir(exist_ok=True)
    done = json.loads(dest.read_text()) if dest.exists() else {}
    with ThreadPoolExecutor(args.workers) as ex:
        for item_id, body in ex.map(lambda w: classify(w, args.judge), [w for w in wrong_items(args.model, args.cond) if w[0] not in done]):
            if body:
                done[item_id] = body
    dest.write_text(json.dumps(done, indent=1))
    print(args.model, args.cond, len(done), {c: sum(v['cause'] == c for v in done.values()) for c in CAUSES})


if __name__ == '__main__':
    main()
