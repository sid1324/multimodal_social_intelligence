"""Repeated-RGB control for the clean-60 forced-choice run on the API models.

Condition rep_nn: the item's RGB grid supplied three times (as many image slots as the maps condition, no new
information), with the same instructions as the other forced-choice conditions. One isolated call per item.

Usage: clean60_repeat.py gpt <model> [<model> ...]   |   clean60_repeat.py claude
"""
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean60_nn as n

c = n.c
REPEAT = ("Three images are provided. All three are the same left-to-right grid of five chronological RGB frames from an egocentric (head-mounted) camera, "
          "captured before the camera wearer's next action. " + c.TASK)
c.CONTEXT['rep_nn'] = REPEAT + n.NO_NONE
c.IMAGES['rep_nn'] = ['rgb', 'rgb', 'rgb']


def main():
    tasks = json.loads((c.OUT / 'tasks.json').read_text())
    if sys.argv[1] == 'gpt':
        workdir = tempfile.mkdtemp(prefix='clean60-empty-')
        jobs = [(m, 'rep_nn', t, workdir) for m in sys.argv[2:] for t in tasks]
        with ThreadPoolExecutor(12) as pool:
            list(pool.map(n.gpt_call, jobs))
        for m in sys.argv[2:]:
            folder = c.OUT / 'results' / n.FOLDER[m]
            print(m, len(list(folder.glob('rep_nn__*.json'))), 'of', len(tasks), 'answered;', len(list(folder.glob('rep_nn__*.failed'))), 'failed')
    else:
        jobs = [(name, 'rep_nn', t) for name in n.ci.MODELS for t in tasks]
        with ThreadPoolExecutor(6) as pool:
            list(pool.map(n.ci.call, jobs))
        folder = c.OUT / 'results/claude_single'
        print('claude', len(list(folder.glob('*__rep_nn__*.json'))), 'of', len(jobs), 'answered;', len(list(folder.glob('*__rep_nn__*.failed'))), 'failed')


if __name__ == '__main__':
    main()
