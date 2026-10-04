"""Persist genuine RGB-to-combined recovery cases as results arrive."""
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded'

def main():
    path=OUT/'recovery-alerts.json';old=json.loads(path.read_text()) if path.exists() else {'alerts':[]};existing={(x['workflow'],x['id'],x['combined_condition']) for x in old['alerts']};found=[]
    qwen=json.loads((OUT/'qwen_results.json').read_text());lookup={(x['id'],x['condition']):x for x in qwen}
    for row in qwen:
        if row['condition'] not in ['all_images','all_descriptions']:continue
        base=lookup.get((row['id'],'rgb_images'))
        if base is None or base['both_correct'] or not row['both_correct']:continue
        controls={c:lookup[(row['id'],c)]['both_correct'] for c in ['repeat3_images','shuffled_all_images','repeat3_descriptions','shuffled_all_descriptions'] if (row['id'],c) in lookup}
        caveat='This condition also adds text; controls and the full sample must be inspected before making a geometry claim.' if row['condition']=='all_descriptions' else 'Images-only recovery; inspect repeated-image and shuffled-map controls and the completed sample before attributing the change to aligned geometry.'
        found.append({'workflow':'Qwen32B','id':row['id'],'combined_condition':row['condition'],'baseline_condition':'rgb_images','baseline_action_idx':base['action_original_idx'],'baseline_justification_idx':base['justification_original_idx'],'combined_action_idx':row['action_original_idx'],'combined_justification_idx':row['justification_original_idx'],'controls_joint_correct':controls,'caveat':caveat})
    followup=OUT/'gpt56-followup-results.json'
    if followup.exists():
        rows=json.loads(followup.read_text())['items'];agent_lookup={(x['id'],x['condition']):x for x in rows};baselines={x['id']:x for x in json.loads((OUT/'gpt56-next10-rgb-results.json').read_text())['items']};annotations={x['id']:x['annotation'] for x in json.loads((ROOT/'output/idea5-audit/protocol.json').read_text())['sample']}
        def score(r):
            a=permutation(r['id'],'action')[r['action_position']-1];j=permutation(r['id'],'justification')[r['justification_position']-1];return a==j==annotations[r['id']]['correct'],a,j
        for row in rows:
            if row['condition'] not in ['all_images','all_descriptions']:continue
            good,a,j=score(row);base=baselines[row['id']];bg,ba,bj=score(base)
            if bg or not good:continue
            controls={c:score(agent_lookup[(row['id'],c)])[0] for c in ['repeat3_images','shuffled_all_images','repeat3_descriptions','shuffled_all_descriptions'] if (row['id'],c) in agent_lookup}
            found.append({'workflow':'GPT56_agent','id':row['id'],'combined_condition':row['condition'],'baseline_condition':'rgb_images','baseline_action_idx':ba,'baseline_justification_idx':bj,'combined_action_idx':a,'combined_justification_idx':j,'controls_joint_correct':controls,'caveat':'Outcome-selected failure diagnostic in a tool-using, accumulating agent context; not an independent causal or stateless model comparison.'})
    alerts=[];new=0
    for row in found:
        key=(row['workflow'],row['id'],row['combined_condition']);previous=next((x for x in old['alerts'] if (x['workflow'],x['id'],x['combined_condition'])==key),None)
        row['first_detected_at']=previous['first_detected_at'] if previous else datetime.now(timezone.utc).isoformat();alerts.append(row)
        if key not in existing:print('NEW RGB-TO-COMBINED RECOVERY: '+json.dumps(row),flush=True);new+=1
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps({'baseline':'Plain RGB images only; require both action and justification correct after intervention','qwen_rows_checked':len(qwen),'alerts':alerts},indent=2));tmp.replace(path)
    return new

if __name__=='__main__':main()
