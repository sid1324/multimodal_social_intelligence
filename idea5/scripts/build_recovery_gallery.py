"""Display saved recovery examples with exact inputs and all current controls."""
import html,json,os
from pathlib import Path
from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded'

def main():
    alerts=json.loads((OUT/'recovery-alerts.json').read_text())['alerts'];inputs={x['id']:x for x in json.loads((OUT/'inputs.json').read_text())['items']};annotations={x['id']:x['annotation'] for x in json.loads((ROOT/'output/idea5-audit/protocol.json').read_text())['sample']};qwen=json.loads((OUT/'qwen_results.json').read_text());esc=html.escape;cards=[]
    for alert in alerts:
        i=alert['id'];item=inputs[i];ann=annotations[i];pictures=[]
        for role in ['rgb','pose','segmentation','depth']:pictures.append(f"<article><h3>{role}</h3><a href='{item['images'][role]}'><img src='{item['images'][role]}'></a><p>{esc(item['descriptions'][role])}</p></article>")
        evidence=''
        if alert['workflow']=='Qwen32B':
            rows=[]
            for r in qwen:
                if r['id']!=i:continue
                a=r['action_original_idx'];j=r['justification_original_idx'];action='Invalid' if a is None else ann['behaviors'][a] or 'None of the offered actions';just='Invalid' if j is None else ann['justifications'][j] or 'None of the offered justifications'
                rows.append(f"<tr><td>{esc(r['condition'])}</td><td>{esc(action)}</td><td>{esc(just)}</td><td>{'Correct' if r['both_correct'] else 'Wrong'}</td></tr>")
            evidence=f"<table><tr><th>Condition</th><th>Action</th><th>Justification</th><th>Both</th></tr>{''.join(rows)}</table>"
        else:
            baseline=json.loads((OUT/'gpt56-next10-rgb-results.json').read_text())['items']
            followup=json.loads((OUT/'gpt56-followup-results.json').read_text())['items']
            rows=[]
            for r in baseline+followup:
                if r['id']!=i:continue
                a=permutation(i,'action')[r['action_position']-1];j=permutation(i,'justification')[r['justification_position']-1]
                condition=r.get('condition','rgb_images');correct=a==j==ann['correct']
                rows.append(f"<tr><td>{esc(condition)}</td><td>{esc(ann['behaviors'][a] or 'None of the offered actions')}</td><td>{esc(ann['justifications'][j] or 'None of the offered justifications')}</td><td>{'Correct' if correct else 'Wrong'}</td></tr>")
            evidence=f"<p>The follow-up choices were confirmed/revised after additional visual inspection; see the audit in the raw results. <a href='gpt56-failures.html'>Complete GPT failure study</a>.</p><table><tr><th>Condition</th><th>Action</th><th>Justification</th><th>Both</th></tr>{''.join(rows)}</table>"
        cards.append(f"<section><h2>{esc(alert['workflow'])}: {esc(i)}</h2><p>RGB baseline wrong; <strong>{esc(alert['combined_condition'])} gets both right</strong>. {esc(alert['caveat'])}</p><p><strong>Reference action:</strong> {esc(ann['behaviors'][ann['correct']])}<br><strong>Reference justification:</strong> {esc(ann['justifications'][ann['correct']])}</p><div class='grid'>{''.join(pictures)}</div>{evidence}</section>")
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><title>EgoNormia recovery cases</title><style>body{{font:16px/1.5 system-ui;background:#f3f6f9;color:#24384d}}main{{max-width:1200px;margin:25px auto;padding:25px;background:white}}section{{margin:24px 0;border-top:2px solid #177e89}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}article{{background:#f4f7fa;padding:12px}}img{{width:100%}}article p{{font-size:12px;white-space:pre-wrap}}table{{width:100%;border-collapse:collapse;font-size:12px;margin:20px 0}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #d8e0e7}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main><h1>RGB-to-combined recovery cases</h1><p>Live snapshot from saved results. The images-only condition and the images-plus-descriptions condition are distinct. A recovery is a selected observation, not proof of general improvement. Inspect repeated-image and shuffled-map controls, then the completed 40-case results.</p>{''.join(cards) or '<p>No qualifying recovery observed yet.</p>'}</main></body></html>"""
    destination=OUT/'recoveries.html';temporary=destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    temporary.write_text(doc);temporary.replace(destination)

if __name__=='__main__':main()
