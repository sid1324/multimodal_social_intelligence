"""Rate how much each test item's choice depends on spatial layout, from the options alone.

GPT-5.6 sees only the five actions and five justifications (no image, no gold) and rates 0-3 how much choosing
the right pair would depend on knowing distances, positions, facing, reach or paths. One isolated call per item.
Output: output/idea5-midterm/spatial_need.json  {id: {"score": 0-3, "reason": "..."}}

usage: midterm_spatial_need.py [--workers 6]
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
OUT = MID / 'spatial_need.json'
PROMPT = ('Below are the candidate actions and justifications for a question about what a camera wearer should do next '
          'in an egocentric video. You cannot see the video. Rate how much choosing the correct action and justification '
          'would depend on knowing the spatial layout of the scene (distances between people, who is in whose path or '
          'personal space, which way people face, what is within reach): 0 = not at all, 1 = slightly, 2 = substantially, '
          '3 = it is mainly a spatial judgement. Return only JSON: {"score": <0-3>, "reason": "<one sentence>"}\n\n')


def rate(task):
    text = PROMPT + 'Actions:\n' + '\n'.join(f'- {a}' for a in task['action_options']) + '\n\nJustifications:\n' + \
        '\n'.join(f'- {j}' for j in task['justification_options'])
    for _ in range(3):
        with tempfile.TemporaryDirectory(prefix='need_') as tmp:
            out = Path(tmp) / 'reply.txt'
            subprocess.run(['codex', 'exec', '-m', 'gpt-5.6-sol', '--sandbox', 'read-only', '--skip-git-repo-check',
                            '--ephemeral', '-C', tmp, '-o', str(out), '-'], input=text, capture_output=True, text=True,
                           timeout=600)
            reply = out.read_text() if out.exists() else ''
        try:
            body = json.loads(reply[reply.find('{'):reply.rfind('}') + 1])
            if body.get('score') in (0, 1, 2, 3):
                return task['id'], body
        except ValueError:
            continue
    return task['id'], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    tasks = json.loads((MID / 'lora_pilot/egonormia_tasks.json').read_text())
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    with ThreadPoolExecutor(args.workers) as ex:
        for item_id, body in ex.map(rate, [t for t in tasks if t['id'] not in done]):
            if body:
                done[item_id] = body
    OUT.write_text(json.dumps(done, indent=1))
    print('rated', len(done), {s: sum(v['score'] == s for v in done.values()) for s in range(4)})


if __name__ == '__main__':
    main()
