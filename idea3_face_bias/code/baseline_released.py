"""Do VLMs already lean toward or away from person-directed actions? Uses EgoNormia's released
per-item answers (src/final_dataset/final_data_eval.json at the pinned annotation commit).

For each wrong answer (chosen and gold options both non-empty), compare whether the chosen option is
person-directed with the rate expected if the model had picked uniformly among that item's
non-gold options. Positive = leans toward engaging the other person.
95% CIs: 5,000 bootstrap replicates clustered by source video ID.
Writes results/released_baseline.json.
"""
import json, collections as C
from pathlib import Path
import numpy as np
from text_arm import person_directed

HERE = Path(__file__).parent
FULL_INPUT = ['QwenVL-25', 'gpt-4o-240513', 'gemini-15-pro-002', 'gemini-15-flash-002', 'gemini-20-flash-exp',
              'gemini-2.5-flash-preview-04-17', 'intern', 'meta/llama-32-90b-vision-instruct-maas']
B, SEED = 5000, 7


def main():
    ev = json.loads((HERE / 'data/released_predictions.json').read_text())
    ann = json.loads((HERE / 'data/official_annotations.json').read_text())
    rng = np.random.default_rng(SEED)
    out = {}
    for m in FULL_INPUT:
        by_src = C.defaultdict(list)  # source id -> list of (observed, expected)
        answered = correct = 0
        for k, v in ev.items():
            if m not in v:
                continue
            beh, gold = ann[k]['behaviors'], ann[k]['correct']
            r = v[m]['best']['results'][0]
            if not beh[gold].strip() or not isinstance(r, int) or not 0 <= r < len(beh):
                continue
            answered += 1
            if r == gold:
                correct += 1
                continue
            others = [i for i in range(len(beh)) if i != gold and beh[i].strip()]
            if not beh[r].strip() or not others:
                continue
            by_src[k.split('_')[0]].append((person_directed(beh[r]),
                                             np.mean([person_directed(beh[i]) for i in others])))
        if answered < 1000:
            continue
        groups = [np.array(g, float) for g in by_src.values()]
        allv = np.concatenate(groups)
        diff = float(allv[:, 0].mean() - allv[:, 1].mean())
        boots = []
        for _ in range(B):
            pick = rng.integers(0, len(groups), len(groups))
            s = np.concatenate([groups[i] for i in pick])
            boots.append(s[:, 0].mean() - s[:, 1].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {'answered': answered, 'accuracy': round(correct / answered, 3), 'wrong_scored': len(allv),
                  'observed_pct': round(100 * allv[:, 0].mean(), 1), 'expected_pct': round(100 * allv[:, 1].mean(), 1),
                  'diff_pp': round(100 * diff, 1), 'ci95_pp': [round(100 * lo, 1), round(100 * hi, 1)]}
        print(f"{m:40s} n={answered:4d} acc={correct/answered:.3f} wrong={len(allv):4d} "
              f"obs={100*allv[:,0].mean():.1f} exp={100*allv[:,1].mean():.1f} diff={100*diff:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")
    (HERE / 'results/released_baseline.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
