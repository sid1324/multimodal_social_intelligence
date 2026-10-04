"""Score completed new RGB predictions externally and prepare gold-free diagnostics."""
import json
from pathlib import Path
from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded'

def main():
    raw=json.loads((OUT/'gpt56-next10-rgb-results.json').read_text());rows=raw if isinstance(raw,list) else raw.get('items',raw.get('results',[]))
    expected=json.loads((OUT/'gpt56-next10-rgb-blind-manifest.json').read_text())['items'];assert len(rows)==len(expected)==10
    assert {x['id'] for x in rows}=={x['id'] for x in expected}
    annotations={x['id']:x['annotation'] for x in json.loads((ROOT/'output/idea5-audit/protocol.json').read_text())['sample']}
    scored=[]
    for r in rows:
        a=permutation(r['id'],'action')[r['action_position']-1];j=permutation(r['id'],'justification')[r['justification_position']-1];gold=annotations[r['id']]['correct']
        scored.append({'id':r['id'],'action_original_idx':a,'justification_original_idx':j,'gold_original_idx':gold,'action_correct':a==gold,'justification_correct':j==gold,'both_correct':a==gold and j==gold})
    failures={x['id'] for x in scored if not x['both_correct']}
    full=json.loads((OUT/'full-blind-manifest.json').read_text());selected=[]
    for r in full['items']:
        if r['id'] not in failures:continue
        selected.append({**r,'conditions':{c:ev for c,ev in r['conditions'].items() if c!='rgb_images'}})
    (OUT/'gpt56-next10-rgb-scored.json').write_text(json.dumps({'n':10,'scores':{metric:sum(x[metric] for x in scored) for metric in ['action_correct','justification_correct','both_correct']},'items':scored},indent=2))
    (OUT/'gpt56-followup-blind-manifest.json').write_text(json.dumps({'gold_free':True,'items':selected,'indexing':'1-based positions','note':'Additional fixed-option representation conditions; no labels or earlier choices supplied.'},indent=2))
    (OUT/'gpt56-failure-study-protocol.json').write_text(json.dumps({'baseline_selection':'Next ten fixed-order cases (6-15) from the existing source-disjoint 40-clip sample, regardless of outcomes.','diagnostic_selection':'All new RGB baselines with either action or justification incorrect, externally identified only after baseline predictions were saved.','failure_ids':sorted(failures),'baseline_n':10,'failure_n':len(failures),'conditions_per_failure':15,'labels_not_provided_to_agent':True,'limit':'Outcome-conditioned failure cohort is diagnostic, not an unbiased score for new inputs. Agent accumulates context; same-example retesting can be influenced by earlier decisions. Results cannot be treated as clean stateless causal estimates.'},indent=2))
    print(json.dumps({'baseline_n':10,'joint_correct':sum(x['both_correct'] for x in scored),'failures':sorted(failures)},indent=2))

if __name__=='__main__':main()
