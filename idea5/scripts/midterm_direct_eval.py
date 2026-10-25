"""Generative models answer the 200 EgoNormia test items directly, with and without explicit spatial input.

Conditions (all include the RGB strip; options are the shuffled lora_pilot tasks, so scoring reuses its answer key):
  rgb         RGB strip only
  maps        + depth, segmentation and pose strips of the same item
  maps_wrong  + the three strips of a fixed other item from a different source video (same partner as X1)
  desc        + the Gemini spatial descriptions of the RGB strip and of each map (text)
  desc_wrong  + the descriptions of the partner item
  rgb_x4      + the RGB strip three more times (rgb2-4.jpg): extra images with no new content
  rgb_cot     RGB, reason step by step before answering (generic reasoning control)
  rgb_spcot   RGB, first analyse the spatial layout (distances, facing, reach, paths), then answer
  blind       no image, options only (how far text priors alone go)
  desc_only   no image, the item's descriptions only (spatial text without pixels)
  desc_only_rep   an identical second run of desc_only (run-to-run noise baseline)
  desc_only_flip  no image, the same descriptions with only the spatial facts reversed (midterm_flip_desc.py)
  desc_only_para  no image, the same descriptions reworded with every fact kept (control for the rewrite itself)
  rgb_ctx     + EgoNormia's own scene description ("desc": activity, people, relationships, setting); a perception aid
  rgb_ctx_wrong  + the partner item's scene description
  <cond>_forced  <cond> with the "None of these" action and justification removed (never gold on the test set);
                 the other options keep their numbers, so scoring is unchanged
One isolated call per item and condition; the answer key is never in the call's directory. Resumable.
Output: output/idea5-midterm/direct_eval/<model>/<cond>/<id>.json

Local MLX models (all bf16, loaded from the original Hugging Face weights) are the published spatial fine-tunes and
their exact base models, for the before/after comparison: qwen25vl3b -> spatialladder3b, mindcube3b;
qwen25vl7b -> spacer7b, spatialssrl7b.

usage: midterm_direct_eval.py qwen32|gemini|gpt|<local model> --conds rgb,maps,... [--workers N] [--limit N]
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
GRIDS = MID / 'grids'
DESC = MID / 'descriptions'
OUT = MID / 'direct_eval'
MASKS = ['depth', 'segmentation', 'pose']
TASK = ('Choose the single most socially appropriate next action for the camera wearer from the action options, '
        'then choose the justification that best fits the action you chose.')
STRIP = ("rgb.jpg is a left-to-right strip of five chronological frames from an egocentric (head-mounted) camera, "
         "captured before the camera wearer's next action.")
MAPS = ('depth.jpg, segmentation.jpg and pose.jpg are the same five frames rendered as a monocular depth map (brighter is '
        'nearer), an instance segmentation map (coloured masks with class labels) and a body/hand skeleton map (yellow body, '
        'magenta hands; blank where nothing was detected). The maps are automatic and can be wrong.')
CTX = 'Description of the video context (from the dataset):'
NOTES = 'Spatial notes about the frames, generated automatically (they can be wrong):'
LABEL = {'rgb': 'Frames', 'depth': 'Depth map', 'segmentation': 'Segmentation map', 'pose': 'Pose map'}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def partner(item_id, pool):
    """Same rule as midterm_embed.partner, so maps_wrong uses the X1 partner."""
    source = item_id.split('_')[0]
    return sorted((i for i in pool if i.split('_')[0] != source), key=lambda i: digest(f'x1:{item_id}:{i}'))[0]


def context_desc(item_id):
    global _CTX
    if '_CTX' not in globals():
        _CTX = json.loads((ROOT / 'data/EgoNormia/annotations/final_data.json').read_text())
    return _CTX[item_id]['desc'].strip()


def descriptions(item_id):
    rgb = json.loads((DESC / 'rgb' / f'{item_id}.json').read_text())['descriptions']
    maps = json.loads((DESC / 'maps' / f'{item_id}.json').read_text())['descriptions']
    return {**rgb, **maps}


def build(task, cond, pool):
    """Returns (prompt, {filename: source image}). '<cond>_rep' is an identical second run of <cond>."""
    if cond.endswith('_rep') and cond != 'desc_only_rep':
        return build(task, cond[:-4], pool)
    if cond.endswith('_forced'):
        prompt, images = build(task, cond[:-7], pool)
        prompt = re.sub(r'\n\d\. None of these (actions|justifications) is appropriate', '', prompt)
        return prompt.replace('{"action": <1-5>, "justification": <1-5>}',
                              '{"action": <number>, "justification": <number>} using only the numbers listed'), images
    item_id = task['id']
    images = {'rgb.jpg': GRIDS / f'{item_id}_rgb.jpg'}
    context = STRIP
    if cond in ('maps', 'maps_wrong'):
        src = item_id if cond == 'maps' else partner(item_id, pool)
        images.update({f'{k}.jpg': GRIDS / f'{src}_{k}.jpg' for k in MASKS})
        context += ' ' + MAPS
    if cond == 'rgb_x4':
        images.update({f'rgb{n}.jpg': GRIDS / f'{item_id}_rgb.jpg' for n in (2, 3, 4)})
        context += ' rgb2.jpg, rgb3.jpg and rgb4.jpg are copies of rgb.jpg.'
    if cond in ('blind', 'desc_only', 'desc_only_rep', 'desc_only_flip', 'desc_only_para'):
        images, context = {}, 'No video frames are available for this scene.'
    if cond in ('desc', 'desc_wrong', 'desc_only', 'desc_only_rep'):
        d = descriptions(item_id if cond != 'desc_wrong' else partner(item_id, pool))
        context += f'\n\n{NOTES}\n' + '\n'.join(f'- {LABEL[k]}: {d[k]}' for k in ['rgb'] + MASKS)
    if cond in ('rgb_ctx', 'rgb_ctx_wrong'):
        src = item_id if cond == 'rgb_ctx' else partner(item_id, pool)
        context += f'\n\n{CTX}\n' + context_desc(src)
    if cond in ('desc_only_flip', 'desc_only_para'):
        d = json.loads((MID / ('descriptions_flipped' if cond == 'desc_only_flip' else 'descriptions_paraphrased') /
                              f'{item_id}.json').read_text())
        context += f'\n\n{NOTES}\n' + '\n'.join(f'- {LABEL[k]}: {d[k]}' for k in ['rgb'] + MASKS)
    lines = [context, '', TASK, '', 'Action options:'] + [f'{i}. {o}' for i, o in enumerate(task['action_options'], 1)]
    lines += ['', 'Justification options:'] + [f'{i}. {o}' for i, o in enumerate(task['justification_options'], 1)]
    if cond == 'rgb_cot':
        lines += ['', 'Think step by step about the situation first. End your reply with this JSON on its own line: '
                  '{"action": <1-5>, "justification": <1-5>}']
    elif cond == 'rgb_spcot':
        lines += ['', 'First analyse the spatial layout: who is present, how far each person is from the camera wearer, '
                  'which way they face, what is within reach, and who is in whose path. Then use that analysis to choose. '
                  'End your reply with this JSON on its own line: {"action": <1-5>, "justification": <1-5>}']
    else:
        lines += ['', 'Reply with only this JSON and nothing else: {"action": <1-5>, "justification": <1-5>}']
    return '\n'.join(lines), images


def parse(reply):
    # some models copy the template literally ("action": <2>), so the angle brackets are optional
    found = re.findall(r'"action"\s*:\s*["<]?([1-5]).*?"justification"\s*:\s*["<]?([1-5])', reply or '', re.S)
    return (int(found[-1][0]), int(found[-1][1])) if found else (None, None)  # last JSON: reasoning may come first


def call_gemini(prompt, tmp):
    proc = subprocess.run(['agy', '-p', prompt, '--model', 'gemini-4-argon-medium', '--output-format', 'json'],
                          cwd=tmp, capture_output=True, text=True, timeout=600)
    start = proc.stdout.find('{"conversation_id"')
    meta = json.loads(proc.stdout[start:].splitlines()[0])
    return meta['response'], {'status': meta.get('status'), 'denied': meta.get('denied_actions')}


def call_gpt(prompt, tmp, names):
    order = ', '.join(f'image {n + 1} = {name}' for n, name in enumerate(names)) or 'none'
    out = Path(tmp) / 'reply.txt'
    cmd = ['codex', 'exec', '-m', 'gpt-5.6-sol', '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral', '-C', tmp,
           '-o', str(out)] + [a for n in names for a in ('-i', str(Path(tmp) / n))] + ['-']
    proc = subprocess.run(cmd, input=f'The attached images are, in order: {order}.\n\n{prompt}', capture_output=True,
                          text=True, timeout=600)
    return (out.read_text() if out.exists() else ''), {'returncode': proc.returncode}


def run_cloud(model, task, cond, pool):
    dest = OUT / model / cond / f"{task['id']}.json"
    if dest.exists():
        return 'skip'
    try:
        prompt, images = build(task, cond, pool)
    except FileNotFoundError:
        return 'missing'
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix='direct_') as tmp:
            for name, src in images.items():
                shutil.copy(src, Path(tmp) / name)
            t = time.time()
            try:
                reply, meta = call_gemini(prompt, tmp) if model == 'gemini' else call_gpt(prompt, tmp, list(images))
            except Exception as e:  # noqa: BLE001
                reply, meta = '', {'error': f'{type(e).__name__}: {e}'[:300]}
        a, j = parse(reply)
        if a is not None:
            break
    if a is None and ('error' in meta or meta.get('status') not in (None, 'SUCCESS') or 'quota' in (reply or '').lower()):
        return 'fail'  # not saved, so a later run retries it
    dest.write_text(json.dumps({'id': task['id'], 'cond': cond, 'model': model, 'action_position': a, 'justification_position': j,
                                'raw': reply, 'meta': meta, 'seconds': round(time.time() - t, 1), 'attempts': attempt + 1}))
    return 'ok' if a is not None else 'unparsed'


LOCAL = {'qwen32': 'models/Qwen3-VL-32B-Instruct-8bit',
         'qwen25vl3b': 'models/Qwen2.5-VL-3B-Instruct', 'spatialladder3b': 'models/SpatialLadder-3B',
         'mindcube3b': 'models/MindCube-RL-PlainCogMap-FromSFT',
         'qwen25vl7b': 'models/Qwen2.5-VL-7B-Instruct', 'spacer7b': 'models/SpaceR', 'spatialssrl7b': 'models/Spatial-SSRL-7B'}


def run_qwen(tasks, conds, pool, name='qwen32'):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    path = str(ROOT / LOCAL[name])
    model, processor = load(path)
    config = load_config(path)
    for cond in conds:
        (OUT / name / cond).mkdir(parents=True, exist_ok=True)
        for task in tasks:
            dest = OUT / name / cond / f"{task['id']}.json"
            if dest.exists():
                continue
            try:
                prompt, images = build(task, cond, pool)
            except FileNotFoundError:
                continue
            names = list(images)
            if names:
                prompt = f"The images are, in order: {', '.join(names)}.\n\n{prompt}"
            mx.random.seed(42)
            formatted = apply_chat_template(processor, config, prompt, num_images=len(names))
            t = time.time()
            output = generate(model, processor, formatted, image=[str(images[n]) for n in names] or None,
                              max_tokens=700 if cond in ('rgb_cot', 'rgb_spcot') else 32 if name == 'qwen32' else 512,  # RL-tuned models may think first
                              temperature=0., verbose=False)
            reply = output.text if hasattr(output, 'text') else str(output)
            a, j = parse(reply)
            dest.write_text(json.dumps({'id': task['id'], 'cond': cond, 'model': name, 'action_position': a,
                                        'justification_position': j, 'raw': reply, 'seconds': round(time.time() - t, 1)}))
        print('done', cond, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['gemini', 'gpt'] + list(LOCAL))
    ap.add_argument('--conds', default='rgb,maps,maps_wrong,desc,desc_wrong')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    tasks = json.loads((MID / 'lora_pilot/egonormia_tasks.json').read_text())[:args.limit]
    split = json.loads((MID / 'splits.json').read_text())
    pool = [i for s in ('test', 'train', 'train_ext') for i in split[s]]
    conds = args.conds.split(',')
    if args.model in LOCAL:
        return run_qwen(tasks, conds, pool, args.model)
    for cond in conds:
        (OUT / args.model / cond).mkdir(parents=True, exist_ok=True)
    jobs = [(t, c) for c in conds for t in tasks
            if c.removesuffix('_rep').removesuffix('_forced') not in ('desc', 'desc_wrong', 'desc_only', 'desc_only_flip', 'desc_only_para') or (DESC / 'rgb' / f"{t['id']}.json").exists()]
    counts = {}
    with ThreadPoolExecutor(args.workers) as pool_ex:
        for n, r in enumerate(pool_ex.map(lambda tc: run_cloud(args.model, tc[0], tc[1], pool), jobs), 1):
            counts[r] = counts.get(r, 0) + 1
            if n % 25 == 0:
                print(n, len(jobs), counts, flush=True)
    print('done', counts, flush=True)


if __name__ == '__main__':
    main()
