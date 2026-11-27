"""Pilots for the final study (PILOT_PLAN.md, frozen Nov 26, 2026). Gemma 4 12B, base model only, 300 training gate scenes.
  perc    hearing check: repeat the spoken source remark, 30 validation scenes; both the source word and the trait word must appear
  gate    unsteered answers, typed and spoken: no remark, neutral pair (noise floor), source pilot (looks / record / none, 2 wordings,
          low and high), implicit pilot (4 matched appearance pairs, no trait word)
  rating  1 to 7 trust rating for every source and implicit cue, typed and spoken, 100 of the 300 scenes (expected value over digits)
  report  CPU: flips minus floor per cue family and format, looks minus record (paired by scene), ratings, the frozen rules
Usage: python src/gemma_pilot.py STAGE   (resumable; exit 75 = time limit, run again)"""
import json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, CONFIG, load_annotations, load_splits, load_item_table, gate_items, permuted_options, seeded_int
from channels import build, build_repeat, scene, CUES, voice_for
from evaluate import row_for, read_rows
from analyze import boot, pct
import gemma_speech as gs

R = Path(os.environ.get('GP_OUT', str(HERE / 'results/gemma_pilot')))
R.mkdir(parents=True, exist_ok=True)
CP = yaml.safe_load((CONFIG / 'cues_pilot.yaml').read_text())
SOURCES = ('looks', 'record', 'none')
N_W = {s: len(CP['source'][s]['low']) for s in SOURCES}
N_IMP = len(CP['implicit']['low'])
FLOOR = ('L1_neutral', 'L1_neutral_paraphrase')
CHANNELS = ('text', 'speech')
N_GATE = int(os.environ.get('GS_N_GATE', 300))


def family_pairs():
    F = {f'src_{s}': [(f'P_src_{s}_low{i}', f'P_src_{s}_high{i}') for i in range(N_W[s])] for s in SOURCES}
    F['implicit'] = [(f'P_imp_low{i}', f'P_imp_high{i}') for i in range(N_IMP)]
    F['floor'] = [FLOOR]
    return F


def cue_conds():
    return [c for f, ps in family_pairs().items() for p in ps for c in p]


def repair(path):
    gs.repair(path)


# ---- GPU stages --------------------------------------------------------------------------------------------------
def perc():
    m, ann = gs.model(), load_annotations()
    items = sorted(load_splits()['pool']['val'], key=lambda k: seeded_int('perc-pilot-v1', k))[:30]
    conds = [f'P_src_{s}_{side}{i}' for s in ('looks', 'record') for side in ('low', 'high') for i in range(N_W[s])]
    SRCW = {'looks': ('looks', 'appearance'), 'record': ('record', 'past actions')}
    out = R / 'perception.jsonl'; repair(out)
    done = {r['id'] for r in read_rows(out)}
    norm = lambda s: ' '.join(re.sub(r'[^a-z ]', ' ', s.lower()).split())
    with open(out, 'a') as fh:
        for j, cid in enumerate(items):
            if cid in done:
                continue
            if gs.timeup():
                sys.exit(75)
            cond = conds[j % len(conds)]; src = cond.split('_')[2]; cue = CUES[cond]
            trait = next(w for w in ('untrustworthy', 'trustworthy', 'cannot be trusted', 'can be trusted') if w in cue)
            g = m.generate(build_repeat(ann[cid], scene(cid), cond, 'speech'), max_new_tokens=60)
            t = norm(re.sub(r'<[^>]*>', ' ', g))
            hit_src = any(re.search(r'\b' + norm(w) + r'\b', t) for w in SRCW[src])
            hit_trait = re.search(r'\b' + norm(trait) + r'\b', t) is not None
            fh.write(json.dumps({'id': cid, 'cond': cond, 'voice': voice_for(cid), 'cue': cue, 'gen': g, 'hit_source': hit_src,
                                 'hit_trait': hit_trait, 'hit': hit_src and hit_trait}) + '\n'); fh.flush()
            print('speech', hit_src, hit_trait, '|', g[:140].replace('\n', ' '), flush=True)
    rows = read_rows(out)
    S = {'speech': float(np.mean([r['hit'] for r in rows])), 'source_word': float(np.mean([r['hit_source'] for r in rows])),
         'trait_word': float(np.mean([r['hit_trait'] for r in rows]))}
    for vo in ('female', 'male'):
        S[f'speech {vo}'] = float(np.mean([r['hit'] for r in rows if r['voice'] == vo] or [float('nan')]))
    S['below_0.90'] = S['speech'] < 0.9
    (R / 'perception.json').write_text(json.dumps(S, indent=1)); print('PERCEPTION', S, flush=True)


def gate():
    m, ann = gs.model(), load_annotations()
    items = gate_items(load_splits(), load_item_table(), 300)[:N_GATE]
    cells = [('none', 'text')] + [(c, ch) for ch in CHANNELS for c in cue_conds()]
    out = R / 'gate_rows.jsonl'; repair(out)
    done = defaultdict(set)
    for r in read_rows(out):
        done[r['id']].add((r['condition'], r['channel']))
    t0 = time.time()
    with open(out, 'a') as fh:
        for i, cid in enumerate(items):
            todo = [c for c in cells if c not in done[cid]]
            if not todo:
                continue
            if gs.timeup():
                print(f'gate: time limit at scene {i}/{len(items)}', flush=True); sys.exit(75)
            v, img = ann[cid], scene(cid)
            outs, _ = m.score_many([build(v, img, c, ch) for c, ch in todo], len(permuted_options(v)))
            fh.write(''.join(json.dumps(row_for(cid, v, c, ch, r, 'train', 'base', gs.MODEL)) + '\n' for (c, ch), r in zip(todo, outs))); fh.flush()
            if (i + 1) % 10 == 0:
                print(f'gate: {i + 1}/{len(items)} scenes, {(time.time() - t0) / (i + 1):.1f} s/scene', flush=True)
    print('gate finished', flush=True)


def rating_prompt(v, img, cond, channel, reversed_scale=False):
    parts = build(v, img, cond, channel)
    q, out = CP['rating_question_reversed' if reversed_scale else 'rating_question'], []
    for k, x in parts:
        if k == 'text' and 'What should the camera wearer do next?' in x:
            x = x[:x.index('What should the camera wearer do next?')] + q
        out.append((k, x))
    return out


def rating():
    import torch
    m, ann = gs.model(), load_annotations()
    items = gate_items(load_splits(), load_item_table(), 300)[:min(100, N_GATE)]
    conds = [c for f, ps in family_pairs().items() if f != 'floor' for p in ps for c in p] + ['L1_neutral']
    digits = [m.tok.encode(str(d), add_special_tokens=False) for d in range(1, 8)]
    assert all(len(d) == 1 for d in digits), digits
    ids = [d[0] for d in digits]
    out = R / 'ratings.jsonl'; repair(out)
    done = {(r['id'], r['condition'], r['channel']) for r in read_rows(out)}
    with open(out, 'a') as fh:
        for i, cid in enumerate(items):
            todo = [(c, ch) for ch in CHANNELS for c in conds if (cid, c, ch) not in done]
            if not todo:
                continue
            if gs.timeup():
                sys.exit(75)
            v, img = ann[cid], scene(cid)
            buf = ''
            for c, ch in todo:
                ev, mass = [], []
                for rev in (False, True):   # counterbalanced scale (cues_pilot.yaml)
                    with torch.no_grad():
                        hid, _ = m.last_hidden([rating_prompt(v, img, c, ch, rev)])
                        lg = m.logits32(hid[0])
                    full = torch.log_softmax(lg, -1)
                    p = torch.softmax(lg[ids], -1)
                    ev.append(float((p * torch.arange(1, 8, device=p.device, dtype=p.dtype)).sum())); mass.append(float(full[ids].exp().sum()))
                buf += json.dumps({'id': cid, 'condition': c, 'channel': ch, 'rating': (ev[0] + 8 - ev[1]) / 2, 'rating_forward': ev[0],
                                   'rating_reversed_flipped': 8 - ev[1], 'digit_mass': float(np.mean(mass))}) + '\n'
            fh.write(buf); fh.flush()
            if (i + 1) % 10 == 0:
                print(f'rating: {i + 1}/{len(items)} scenes', flush=True)
    print('rating finished', flush=True)


# ---- CPU stage -----------------------------------------------------------------------------------------------------
def paired_boot(per, common):
    byv = defaultdict(list)
    for c in common:
        byv[c.split('_')[0]].append(per[c])
    return boot(byv)


def report():
    rows = read_rows(R / 'gate_rows.jsonl')
    P = defaultdict(dict)
    for r in rows:
        P[r['id']][(r['condition'], r['channel'])] = r['pred']
    F = family_pairs(); S = {}; md = ['# Pilots (Gemma 4 12B, 300 training scenes, base model)', '']
    if (R / 'perception.json').exists():
        pj = json.loads((R / 'perception.json').read_text()); S['perception'] = pj
        md += [f"Hearing check (spoken source remark repeated with both the source word and the trait word, 30 scenes): {pj['speech']:.2f} "
               f"(source word {pj['source_word']:.2f}, trait word {pj['trait_word']:.2f}; female {pj.get('speech female', float('nan')):.2f}, "
               f"male {pj.get('speech male', float('nan')):.2f}). Target 0.90{' not met (recorded, not a stop)' if pj['below_0.90'] else ' met'}.", '']
    md += ['| Format | Cue family | n | Flips | Flips minus floor |', '|---|---|---|---|---|']
    per = {}
    for ch in CHANNELS:
        flips = {}
        for f, ps in F.items():
            flips[f] = {cid: float(np.mean([d[(a, ch)] != d[(b, ch)] for a, b in ps])) for cid, d in P.items()
                        if all((a, ch) in d and (b, ch) in d for a, b in ps)}
        per[ch] = flips
        common = set.intersection(*[set(x) for x in flips.values()])
        for f in F:
            if f == 'floor':
                continue
            o = {'n': len(common), 'flips': paired_boot(flips[f], common),
                 'minus_floor': paired_boot({c: flips[f][c] - flips['floor'][c] for c in common}, common)}
            S[f'{ch} {f}'] = o
            md.append(f"| {ch} | {f} | {o['n']} | {pct(o['flips'])} | {pct(o['minus_floor'])} |")
        S[f'{ch} floor'] = {'flips': paired_boot(flips['floor'], common)}
        S[f'{ch} looks minus record'] = paired_boot({c: flips['src_looks'][c] - flips['src_record'][c] for c in common}, common)
        md.append(f"| {ch} | floor (neutral rewording) | {len(common)} | {pct(S[f'{ch} floor']['flips'])} | |")
    md += ['', '| Format | Looks flips minus record flips (paired) |', '|---|---|'] + [f"| {ch} | {pct(S[f'{ch} looks minus record'])} |" for ch in CHANNELS]
    # ratings
    rr = read_rows(R / 'ratings.jsonl')
    if rr:
        Q = defaultdict(dict)
        for r in rr:
            Q[r['id']][(r['condition'], r['channel'])] = r['rating']
        md += ['', '## Trust ratings (1 to 7; low cue minus high cue, paired by scene)', '', '| Format | Cue family | Low minus high | Pairs in the expected direction |', '|---|---|---|---|']
        for ch in CHANNELS:
            for f, ps in F.items():
                if f == 'floor':
                    continue
                d = {cid: float(np.mean([q[(a, ch)] - q[(b, ch)] for a, b in ps])) for cid, q in Q.items() if all((a, ch) in q and (b, ch) in q for a, b in ps)}
                if not d:
                    continue
                pair_means = [float(np.mean([q[(a, ch)] - q[(b, ch)] for q in Q.values() if (a, ch) in q and (b, ch) in q])) for a, b in ps]
                o = {'n': len(d), 'low_minus_high': paired_boot(d, set(d)), 'pairs_negative': sum(x < 0 for x in pair_means), 'n_pairs': len(ps)}
                S[f'rating {ch} {f}'] = o
                lm = o['low_minus_high']
                md.append(f"| {ch} | {f} | {lm[0]:.2f} [{lm[1]:.2f}, {lm[2]:.2f}] | {o['pairs_negative']} of {o['n_pairs']} |")
            S[f'rating {ch} digit_mass'] = float(np.mean([r['digit_mass'] for r in rr if r['channel'] == ch]))
            fw = [r['rating_forward'] for r in rr if r['channel'] == ch]; rv = [r['rating_reversed_flipped'] for r in rr if r['channel'] == ch]
            S[f'rating {ch} scale agreement r'] = float(np.corrcoef(fw, rv)[0, 1]) if len(fw) > 2 and np.std(fw) > 0 and np.std(rv) > 0 else None
        md += ['', f"Mean probability on the digits 1 to 7: typed {S.get('rating text digit_mass', float('nan')):.3f}, spoken {S.get('rating speech digit_mass', float('nan')):.3f}.",
               f"Agreement between the two scale directions (correlation): typed {S.get('rating text scale agreement r')}, spoken {S.get('rating speech scale agreement r')}. Low agreement means the ratings are not trustworthy."]
    # frozen rules (PILOT_PLAN.md)
    above = lambda k: bool(S[k]['minus_floor'][1] > 0)
    p1 = {'(b) record above floor, typed': above('text src_record'), '(b) record above floor, spoken': above('speech src_record'),
          '(c) looks above floor, typed': above('text src_looks')}
    p1_go = all(p1.values())
    looks_spoken = above('speech src_looks')
    rt = S.get('rating text implicit')
    p2 = {'manipulation check (typed): low minus high <= -1.0, range below 0, >= 3 of 4 pairs': bool(rt and rt['low_minus_high'][0] <= -1.0 and rt['low_minus_high'][2] < 0 and rt['pairs_negative'] >= 3),
          'action effect (typed): implicit flips above floor': above('text implicit')}
    S['pilot1_rules'] = p1; S['pilot1_go'] = p1_go; S['looks_spoken_above_floor'] = looks_spoken
    S['pilot2_rules'] = p2; S['pilot2_include'] = all(p2.values())
    md += ['', '## Frozen rules (PILOT_PLAN.md)', '', '**Pilot 1 (source control):**'] + [f'- {k}: **{"yes" if v else "no"}**' for k, v in p1.items()]
    md += [f"- Looks above floor, spoken (decides whether speech has a looks effect to fix): **{'yes' if looks_spoken else 'no'}**",
           f"- **Pilot 1: {'GO, the looks vs record contrast becomes the main measure' if p1_go else 'NO GO, behavior sentences stay the primary check; record cues exploratory'}**",
           '', '**Pilot 2 (implicit, exploratory):**'] + [f'- {k}: **{"yes" if v else "no"}**' for k, v in p2.items()]
    md += [f"- **Pilot 2: {'include in the final study' if S['pilot2_include'] else 'one line: no implicit appearance effect'}**", '']
    (R / 'report.json').write_text(json.dumps(S, indent=1, default=bool)); (R / 'report.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
