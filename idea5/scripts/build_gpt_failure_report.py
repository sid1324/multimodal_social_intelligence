"""Externally score and report an outcome-conditioned GPT agent diagnostic."""
import html
import json
from collections import Counter
from pathlib import Path
from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded'

def main():
    manifest=json.loads((OUT/'gpt56-followup-blind-manifest.json').read_text());cases=manifest['items'];expected={(x['id'],c) for x in cases for c in x['conditions']}
    raw=json.loads((OUT/'gpt56-followup-results.json').read_text());rows=raw['items'];assert len(rows)==len(expected)==60 and {(x['id'],x['condition']) for x in rows}==expected
    private={x['id']:x['annotation'] for x in json.loads((ROOT/'output/idea5-audit/protocol.json').read_text())['sample']}
    original=json.loads((OUT/'gpt56-agent-results.json').read_text())['items'];baseline_rows=json.loads((OUT/'gpt56-next10-rgb-results.json').read_text())['items'];baseline={x['id']:x for x in baseline_rows}
    inputs={x['id']:x for x in json.loads((OUT/'inputs.json').read_text())['items']};cases_by_id={x['id']:x for x in cases}
    def score(row):
        a=row['action_position'];j=row['justification_position'];assert 1<=a<=5 and 1<=j<=5
        ai=permutation(row['id'],'action')[a-1];ji=permutation(row['id'],'justification')[j-1];gold=private[row['id']]['correct']
        return {'id':row['id'],'condition':row.get('condition','rgb'),'action_original_idx':ai,'justification_original_idx':ji,'gold_original_idx':gold,'action_correct':ai==gold,'justification_correct':ji==gold,'both_correct':ai==gold and ji==gold}
    all_baseline=[score(x) for x in original if x['condition']=='rgb']+[score(x) for x in baseline_rows];assert len(all_baseline)==15
    assert sum(x['both_correct'] for x in all_baseline)==11
    scored=[];views=0
    for row in rows:
        ev=cases_by_id[row['id']]['conditions'][row['condition']];assert [x['path'] for x in row['image_view_evidence']]==ev['images']
        assert all(x['viewed'] and Path(x['path']).exists() for x in row['image_view_evidence']);views+=len(row['image_view_evidence'])
        pointer=row['attached_text_source'];node=json.loads(Path(pointer['manifest_path']).read_text())
        for k in pointer['json_pointer'].strip('/').split('/'):node=node[int(k)] if isinstance(node,list) else node[k]
        assert node==ev['attached_text']
        s=score(row);old=score(baseline[row['id']]);assert not old['both_correct'];s['action_changed']=s['action_original_idx']!=old['action_original_idx'];s['justification_changed']=s['justification_original_idx']!=old['justification_original_idx'];scored.append(s)
    by={(x['id'],x['condition']):x for x in scored};conditions=list(cases[0]['conditions']);summary={'rgb_baseline':{'n':15,'action_correct':sum(x['action_correct'] for x in all_baseline),'justification_correct':sum(x['justification_correct'] for x in all_baseline),'both_correct':11},'new10_baseline':{'n':10,'both_correct':6},'diagnostic_n':4,'conditions':{},'limits':'Selected RGB failures, tool-using agent and accumulating context; not unbiased population accuracy, a stateless API benchmark or causal geometry evidence.'}
    for c in conditions:
        r=[x for x in scored if x['condition']==c];summary['conditions'][c]={'n':4,'action_correct':sum(x['action_correct'] for x in r),'justification_correct':sum(x['justification_correct'] for x in r),'both_correct':sum(x['both_correct'] for x in r),'action_changed':sum(x['action_changed'] for x in r),'justification_changed':sum(x['justification_changed'] for x in r)}
    controls=[]
    for mode in ['images','descriptions']:
        for reference in ['repeat3','shuffled_all']:
            aa=f'all_{mode}';bb=f'{reference}_{mode}';improved=sum(by[(x['id'],aa)]['both_correct'] and not by[(x['id'],bb)]['both_correct'] for x in cases);worsened=sum(not by[(x['id'],aa)]['both_correct'] and by[(x['id'],bb)]['both_correct'] for x in cases)
            controls.append({'condition':aa,'reference':bb,'improved':improved,'worsened':worsened,'unchanged':4-improved-worsened})
    summary['matched_controls']=controls
    (OUT/'gpt56-followup-scored.json').write_text(json.dumps(scored,indent=2));(OUT/'gpt56-failure-summary.json').write_text(json.dumps(summary,indent=2))
    esc=html.escape
    metric_rows=''.join(f"<tr><td>{esc(c)}</td><td>{v['action_correct']}/4</td><td>{v['justification_correct']}/4</td><td>{v['both_correct']}/4</td><td>{v['action_changed']}/4</td><td>{v['justification_changed']}/4</td></tr>" for c,v in summary['conditions'].items())
    controls_html=''.join(f"<li>{esc(c['condition'])} vs {esc(c['reference'])}: {c['improved']} improved, {c['worsened']} worsened, {c['unchanged']} unchanged joint outcomes.</li>" for c in controls)
    details=[]
    for case in cases:
        i=case['id'];ann=private[i];old=score(baseline[i]);picture=[]
        for role in ['rgb','pose','segmentation','depth']:
            picture.append(f"<article><h3>{role}</h3><a href='{inputs[i]['images'][role]}'><img src='{inputs[i]['images'][role]}' loading='lazy'></a><p>{esc(inputs[i]['descriptions'][role])}</p></article>")
        def text(idx,key):return 'Invalid' if idx is None else ann[key][idx] or 'None of the offered options'
        case_rows=[]
        for c in conditions:
            s=by[(i,c)];ev=case['conditions'][c];links=' '.join(f"<a href='{Path(path).relative_to(OUT)}'>Image{k+1}</a>" for k,path in enumerate(ev['images']));case_rows.append(f"<tr><td>{esc(c)}<details><summary>Exact inputs</summary>{links}<pre>{esc(ev['attached_text'])}</pre></details></td><td>{esc(text(s['action_original_idx'],'behaviors'))}</td><td>{esc(text(s['justification_original_idx'],'justifications'))}</td><td>{'Correct' if s['both_correct'] else 'Wrong'}</td></tr>")
        details.append(f"<details><summary>{esc(i)}</summary><p><strong>RGB baseline action:</strong> {esc(text(old['action_original_idx'],'behaviors'))}<br><strong>RGB baseline justification:</strong> {esc(text(old['justification_original_idx'],'justifications'))}</p><div class='images'>{''.join(picture)}</div><table><tr><th>Condition</th><th>Action</th><th>Justification</th><th>Both</th></tr>{''.join(case_rows)}</table><details><summary>Reference answers</summary><p>{esc(text(ann['correct'],'behaviors'))}</p><p>{esc(text(ann['correct'],'justifications'))}</p></details></details>")
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><title>GPT agent EgoNormia failure diagnostics</title><style>body{{font:16px/1.5 system-ui;color:#24384d;background:#f3f6f9}}main{{max-width:1150px;margin:25px auto;background:white;padding:28px}}.note{{padding:16px;background:#fff2d6}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #d8e0e7}}th{{background:#edf3f7}}details{{border:1px solid #d8e0e7;padding:12px;margin:12px 0}}summary{{cursor:pointer;font-weight:600}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}article{{background:#f4f7fa;padding:12px}}article img{{width:100%}}article p,pre{{white-space:pre-wrap;font-size:12px}}@media(max-width:700px){{.images{{grid-template-columns:1fr}}}}</style></head><body><main><h1>GPT-5.6-sol-configured agent: failure diagnostics</h1><p>RGB baseline: <strong>11/15 both correct</strong>; the first five were5/5 and the next ten6/10. Action {summary['rgb_baseline']['action_correct']}/15; justification {summary['rgb_baseline']['justification_correct']}/15.</p><div class='note'>This is a tool-using, context-accumulating agent workflow. The requested model override was accepted but not independently inspected inside the agent. The four follow-up cases were selected because their RGB baseline failed; follow-up scores are recovery counts on that selected cohort, not general benchmark accuracy. Reference labels were kept out of agent inputs.</div><h2>Recovery on four selected RGB failures</h2><table><tr><th>Condition</th><th>Action</th><th>Justification</th><th>Both/recovered</th><th>Action changed</th><th>Justification changed</th></tr>{metric_rows}</table><h2>Matched controls</h2><ul>{controls_html}</ul><p>Recovery that also appears with repeated RGB or shuffled maps does not isolate a geometry effect. Earlier decisions, extra viewing, or accumulated agent context may influence follow-up choices. No training was performed.</p><h2>Images, attached descriptions and decisions</h2>{''.join(details)}<h2>Disclosure</h2><p>Codex designed and implemented this exploratory workflow, scored predictions externally, and generated this report. Pretrained models generated derived images; Qwen generated RGB captions. Student interpretation, representation review and TA approval remain pending. Raw predictions and view-evidence records are in gpt56-next10-rgb-results.json and gpt56-followup-results.json.</p></main></body></html>"""
    (OUT/'gpt56-failures.html').write_text(doc)
    lines=['# GPT agent failure diagnostic','',f"RGB baseline: 11/15 jointly correct; action {summary['rgb_baseline']['action_correct']}/15, justification {summary['rgb_baseline']['justification_correct']}/15.",'','Follow-up cohort: four selected RGB failures. Not an unbiased benchmark score.','', '| Condition | Joint recovery |','|---|---|']+[f"| {c} | {v['both_correct']}/4 |" for c,v in summary['conditions'].items()]
    lines+=['','Matched controls:']+[f"- {c['condition']} vs {c['reference']}: {c['improved']} improved / {c['worsened']} worsened / {c['unchanged']} unchanged." for c in controls]+['','Context accumulation, repeated viewing and outcome-based selection prevent a clean causal geometry interpretation. AI-generated report; student interpretation pending.']
    (OUT/'gpt56-failure-report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'gpt56-failure-verification.json').write_text(json.dumps({'new_baseline_rows':10,'original_baseline_rows':5,'followup_rows':60,'selected_failures':4,'conditions_per_failure':15,'exact_image_view_records_checked':views,'attached_description_pointers_verified':True,'choices_scored_externally':True},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
