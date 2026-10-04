"""Part 3: can a text trait cue plausibly matter? Lexicon-based check of answer options.

An option is 'person-directed' if it refers to another person (lexicon below).
An item has 'room for a trait cue' if its options include both person-directed and
non-person-directed actions, so a cue about the partner could shift the choice.
Lexicon matching is a heuristic; see spot-check labels for its error rate.
"""
import json, re, sys, csv, collections as C
from pathlib import Path

HERE = Path(__file__).parent
ANN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'data' / 'official_annotations.json'
PRIMARY = {'Privacy', 'Coordination/Proactivity', 'Communication/Legibility', 'Cooperation'}
# they/them/their are excluded: in these annotations they often refer to objects.
PERSON = re.compile(r"\b(person|person's|persons|people|someone|somebody|anyone|everyone|others?|"
                    r"him|her|his|hers|partner|friend|friends|child|children|kid|kids|"
                    r"baby|man|woman|men|women|colleague|coworker|co-worker|customer|cashier|worker|workers|"
                    r"teammate|teammates|team|group|family|guest|guests|companion|stranger|player|players|"
                    r"opponent|teacher|student|clerk|staff|neighbor|mother|father|mom|dad|wife|husband|"
                    r"sister|brother|daughter|son|boy|girl|instructor|driver|passenger|client|client's|"
                    r"bricklayers?|vendor|seller|shopkeeper|waiter|server|"
                    # communicative acts imply an addressee
                    r"ask|asks|asking|greet|greeting|thank|apologi[sz]e|discuss|chat|talk|tell|say|"
                    r"conversation|wave|introduce|invite|whisper|explain|inform|compliment|"
                    r"acknowledg\w*)\b", re.I)


def person_directed(text):
    return bool(PERSON.search(text or ''))


def main():
    ann = json.loads(ANN.read_text())
    rows = []
    for k in sorted(ann):
        v = ann[k]
        opts = [(i, b) for i, b in enumerate(v['behaviors']) if b.strip()]
        gold = v['behaviors'][v['correct']]
        norms = set(v['taxonomy'].get(str(v['correct']), []))
        pd = [person_directed(b) for _, b in opts]
        rows.append({
            'id': k,
            'gold_is_none_sentinel': not gold.strip(),
            'gold_person_directed': person_directed(gold),
            'n_options': len(opts),
            'n_person_directed_options': sum(pd),
            'room_for_trait_cue': any(pd) and not all(pd),
            'primary_norm': bool(norms & PRIMARY),
        })
    n = len(rows)
    s = {
        'items': n,
        'gold_person_directed': sum(r['gold_person_directed'] for r in rows),
        'any_option_person_directed': sum(r['n_person_directed_options'] > 0 for r in rows),
        'room_for_trait_cue': sum(r['room_for_trait_cue'] for r in rows),
        'room_and_primary_norm': sum(r['room_for_trait_cue'] and r['primary_norm'] for r in rows),
        'gold_none_sentinel': sum(r['gold_is_none_sentinel'] for r in rows),
    }
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    with open(out / 'text_arm_items.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / 'text_arm_summary.json').write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
