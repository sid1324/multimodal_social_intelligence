"""Readable results/gallery for expanded image and description comparisons."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/idea5-audit';OUT=ROOT/'output/idea5-expanded'
BASES=['rgb','pose','segmentation','depth','all','repeat1','repeat3','shuffled_all']
LABELS={'rgb':'RGB','pose':'RGB + pose','segmentation':'RGB + segmentation','depth':'RGB + depth','all':'RGB + all three','repeat1':'RGB + repeated RGB','repeat3':'RGB repeated four times','shuffled_all':'RGB + all three shuffled'}

def bootstrap(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(42);v=x[rng.integers(0,len(x),size=(5000,len(x)))].mean(axis=1)
    return [float(x.mean()),float(np.quantile(v,.025)),float(np.quantile(v,.975))]

def main():
    inputs=json.loads((OUT/'inputs.json').read_text()); items=inputs['items'];ids=[x['id'] for x in items]
    results=json.loads((OUT/'qwen_results.json').read_text());lookup={(x['id'],x['condition']):x for x in results}
    assert len(lookup)==len(results)
    conditions={f'{b}_{m}' for b in BASES for m in ['images','descriptions']}
    ids=[i for i in ids if all((i,c) in lookup for c in conditions)]
    assert ids
    n=len(ids)
    scores={};differences=[]
    for mode in ['images','descriptions']:
        for base in BASES:
            c=f'{base}_{mode}';scores[c]={metric:{'correct':sum(lookup[(i,c)][metric] for i in ids),'n':len(ids),'ci':bootstrap([lookup[(i,c)][metric] for i in ids])} for metric in ['action_correct','justification_correct','both_correct']}
    comparisons=[(f'{b}_{m}',f'rgb_{m}') for m in ['images','descriptions'] for b in ['pose','segmentation','depth','all']]+[(f'{b}_descriptions',f'{b}_images') for b in ['rgb','pose','segmentation','depth','all']]+[(f'all_{m}',f'{b}_{m}') for m in ['images','descriptions'] for b in ['repeat3','shuffled_all']]
    for a,b in comparisons:
        values=[int(lookup[(i,a)]['both_correct'])-int(lookup[(i,b)]['both_correct']) for i in ids]
        differences.append({'condition':a,'reference':b,'joint_difference_ci':bootstrap(values),'improved':values.count(1),'worsened':values.count(-1),'unchanged':values.count(0),'bootstrap_degenerate':not any(values),
            'action_difference_ci':bootstrap([int(lookup[(i,a)]['action_correct'])-int(lookup[(i,b)]['action_correct']) for i in ids]),
            'justification_difference_ci':bootstrap([int(lookup[(i,a)]['justification_correct'])-int(lookup[(i,b)]['justification_correct']) for i in ids]),
            'action_predictions_changed':sum(lookup[(i,a)]['action_original_idx']!=lookup[(i,b)]['action_original_idx'] for i in ids),
            'justification_predictions_changed':sum(lookup[(i,a)]['justification_original_idx']!=lookup[(i,b)]['justification_original_idx'] for i in ids),
            'either_prediction_changed':sum(any(lookup[(i,a)][key]!=lookup[(i,b)][key] for key in ['action_original_idx','justification_original_idx']) for i in ids)})
    summary={'items':n,'status':'Stopped at user request; partial experiment','saved_rows':len(results),'planned_rows':640,'planned_items':40,'complete_item_ids':ids,'excluded_incomplete_rows':len(results)-n*16,'condition_saved_counts':{c:sum(r['condition']==c for r in results) for c in sorted(conditions)},'source_videos':40,'frames':200,'model':json.loads((OUT/'model.json').read_text()),'scores':scores,'paired_joint_differences':differences,'invalid_outputs':{c:sum(lookup[(i,c)]['action_original_idx'] is None or lookup[(i,c)]['justification_original_idx'] is None for i in ids) for c in scores}}
    agent_path=OUT/'gpt56-agent-results.json'
    agent_scored=[]
    if agent_path.exists():
        agent=json.loads(agent_path.read_text());agent_rows=agent if isinstance(agent,list) else agent.get('items',agent.get('results',agent.get('predictions',[])))
        private={x['id']:x for x in json.loads((OLD/'protocol.json').read_text())['sample']}
        for row in agent_rows:
            item_id=row.get('id',row.get('item_id')); condition=row.get('condition','rgb')
            a=row.get('action_position',row.get('action_choice',row.get('action_index',row.get('action'))));j=row.get('justification_position',row.get('justification_choice',row.get('justification_index',row.get('justification'))))
            if isinstance(a,dict):a=a.get('choice')
            if isinstance(j,dict):j=j.get('choice')
            if not isinstance(a,int) or not isinstance(j,int):continue
            ao=permutation(item_id,'action');jo=permutation(item_id,'justification');aidx=ao[a-1] if 1<=a<=5 else None;jidx=jo[j-1] if 1<=j<=5 else None;gold=private[item_id]['annotation']['correct']
            mapped_condition='rgb_images' if condition=='rgb' else condition
            agent_scored.append({'id':item_id,'condition':mapped_condition,'action_original_idx':aidx,'justification_original_idx':jidx,'gold_original_idx':gold,'action_correct':aidx==gold,'justification_correct':jidx==gold,'both_correct':aidx==gold and jidx==gold})
        agent_scores={}
        for c in sorted({r['condition'] for r in agent_scored}):
            rr=[r for r in agent_scored if r['condition']==c];matched=[lookup[(r['id'],c)] for r in rr if (r['id'],c) in lookup]
            agent_scores[c]={'n':len(rr),'action_correct':sum(r['action_correct'] for r in rr),'justification_correct':sum(r['justification_correct'] for r in rr),'both_correct':sum(r['both_correct'] for r in rr),'qwen_matched_both':sum(r['both_correct'] for r in matched),'qwen_matched_n':len(matched)}
        summary['gpt56_agent_pilot']={'scores':agent_scores,'identity':'Orchestrator accepted requested model override gpt-5.6-sol; not independently inspected inside agent runtime.','protocol_limit':'Tool-using agent with accumulating context; not a clean stateless model/API benchmark. Five items; no significance or model-ranking claim.'}
        (OUT/'gpt56-agent-scored.json').write_text(json.dumps(agent_scored,indent=2))
    if (OUT/'gpt56-failure-summary.json').exists():
        summary['gpt56_failure_study']=json.loads((OUT/'gpt56-failure-summary.json').read_text())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plot(scores)
    build_html(inputs,summary,lookup)
    build_tex(summary)
    print(json.dumps(summary,indent=2))

def plot(scores):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.8));bases=BASES[:5];x=np.arange(len(bases));width=.34
    for offset,mode,label,color in [(-width/2,'images','Images only','#177e89'),(width/2,'descriptions','Images + descriptions','#8c5bb6')]:
        values=np.array([scores[f'{b}_{mode}']['both_correct']['correct']/scores[f'{b}_{mode}']['both_correct']['n']*100 for b in bases]);lo=np.array([scores[f'{b}_{mode}']['both_correct']['ci'][1]*100 for b in bases]);hi=np.array([scores[f'{b}_{mode}']['both_correct']['ci'][2]*100 for b in bases])
        ax.bar(x+offset,values,width,color=color,label=label);ax.errorbar(x+offset,values,yerr=[values-lo,hi-values],fmt='none',ecolor='#263749',capsize=3)
        for k,b in enumerate(bases):ax.text(k+offset,hi[k]+2,f"{scores[f'{b}_{mode}']['both_correct']['correct']}/{scores[f'{b}_{mode}']['both_correct']['n']}",ha='center',fontsize=9)
    ax.set_xticks(x,['RGB','+ Pose','+ Segmentation','+ Depth','+ All three']);ax.set_ylabel('Action and justification both correct (%)');ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.legend(frameon=False,loc='upper right',ncol=2);ax.set_title('Qwen3-VL-32B-Instruct (8-bit): representation and text ablations',loc='left',fontweight='bold',pad=18)
    fig.text(.02,.015,'Matched complete clips only. 95% bootstrap intervals; caption generation sees no options or labels.',fontsize=9,color='#536273');fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(OUT/'comparison.png',dpi=220);plt.close(fig)

def build_html(inputs,s,lookup):
    esc=html.escape;score_rows=[]
    for base in BASES:
        cells=[]
        for mode in ['images','descriptions']:
            metrics=s['scores'][f'{base}_{mode}'];cells.extend(f"<td>{metrics[m]['correct']}/{metrics[m]['n']}</td>" for m in ['action_correct','justification_correct','both_correct'])
        score_rows.append(f"<tr><td>{LABELS[base]}</td>{''.join(cells)}</tr>")
    changes=[]
    for d in s['paired_joint_differences']:
        mean,lo,hi=d['joint_difference_ci'];ci='No observed paired changes; bootstrap degenerates' if d['bootstrap_degenerate'] else f'{mean*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}]'
        changes.append(f"<tr><td>{esc(d['condition'])} vs {esc(d['reference'])}</td><td>{ci}</td><td>{d['improved']}</td><td>{d['worsened']}</td><td>{d['either_prediction_changed']}/{s['items']}</td></tr>")
    agent_section=''
    if 'gpt56_agent_pilot' in s:
        pilot=s['gpt56_agent_pilot'];rows=''.join(f"<tr><td>{esc(c)}</td><td>{v['action_correct']}/{v['n']}</td><td>{v['justification_correct']}/{v['n']}</td><td>{v['both_correct']}/{v['n']}</td><td>{v['qwen_matched_both']}/{v['qwen_matched_n']}</td></tr>" for c,v in pilot['scores'].items())
        agent_section=f"<h2>Requested GPT-5.6-sol agent pilot</h2><p>{esc(pilot['identity'])} {esc(pilot['protocol_limit'])} The agent viewed actual image grids and was not supplied gold labels. Qwen's column uses only the same five examples.</p><table><tr><th>Condition</th><th>Agent action</th><th>Agent justification</th><th>Agent both</th><th>Matched Qwen both</th></tr>{rows}</table>"
    if 'gpt56_failure_study' in s:
        study=s['gpt56_failure_study'];base=study['rgb_baseline'];n=study['diagnostic_n']
        agent_section+=f"<h3>Extended RGB baseline and failure diagnostics</h3><p>Across the first 15 fixed-order clips, the agent gets both answers correct on {base['both_correct']}/{base['n']}. All {n} failures from the new ten-case baseline were retested: all-three images recover {study['conditions']['all_images']['both_correct']}/{n}; all-three with descriptions recover {study['conditions']['all_descriptions']['both_correct']}/{n}. This is an outcome-selected, context-accumulating diagnostic. <a href='gpt56-failures.html'>Inspect the complete failure study, controls, images and descriptions</a>.</p>"
    sample={x['id']:x for x in json.loads((OLD/'protocol.json').read_text())['sample']};details=[]
    for j,item in enumerate(inputs['items']):
        pictures=[]
        for role in ['rgb','pose','segmentation','depth']:
            pictures.append(f"<article><h3>{role.title()}</h3><a href='{item['images'][role]}'><img loading='lazy' src='{item['images'][role]}' alt='{role} sequence grid'></a><p class='caption'>{esc(item['descriptions'][role])}</p></article>")
        rows=''.join(f"<tr><td>{base}</td>"+''.join(f"<td>{('Correct' if lookup[(item['id'],f'{base}_{mode}')]['both_correct'] else 'Wrong') if (item['id'],f'{base}_{mode}') in lookup else 'Not run'}</td>" for mode in ['images','descriptions'])+'</tr>' for base in BASES[:5])
        ann=sample[item['id']]['annotation'];gold=esc(ann['behaviors'][ann['correct']] or 'None of the offered actions');gold_just=esc(ann['justifications'][ann['correct']] or 'None of the offered justifications')
        prediction_rows=[]
        for base in BASES:
            for mode in ['images','descriptions']:
                result=lookup.get((item['id'],f'{base}_{mode}'))
                if result is None:
                    prediction_rows.append(f"<tr><td>{esc(base+' '+mode)}</td><td colspan='3'>Not run</td></tr>");continue
                ai=result['action_original_idx'];ji=result['justification_original_idx']
                action='Invalid response' if ai is None else ann['behaviors'][ai] or 'None of the offered actions'
                justification='Invalid response' if ji is None else ann['justifications'][ji] or 'None of the offered justifications'
                prediction_rows.append(f"<tr><td>{esc(base+' '+mode)}</td><td>{esc(action)}</td><td>{esc(justification)}</td><td>{'Correct' if result['both_correct'] else 'Wrong'}</td></tr>")
        details.append(f"<details><summary>{j+1}. {esc(item['id'])}</summary><p>Each grid has the same five chronological frames. Click an image for full size. Captions are predictions/summaries, not verified annotations.</p><div class='modalities'>{''.join(pictures)}</div><table><tr><th>Input</th><th>Images only, both correct?</th><th>With descriptions, both correct?</th></tr>{rows}</table><details><summary>Inspect every selected action and justification</summary><table><tr><th>Condition</th><th>Selected action</th><th>Selected justification</th><th>Both</th></tr>{''.join(prediction_rows)}</table></details><details><summary>Show reference action and justification</summary><p><strong>Action:</strong> {gold}</p><p><strong>Justification:</strong> {gold_just}</p></details><textarea data-key='{item['id']}' placeholder='Your observations: which maps or descriptions are misleading? What evidence supports the reference action?'></textarea></details>")
    text=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>EgoNormia - expanded 32B experiment</title><style>body{{font:16px/1.55 system-ui,sans-serif;background:#f3f6f9;color:#22354a;margin:0}}main{{max-width:1250px;margin:28px auto;padding:28px;background:white;border-radius:12px}}h1,h2{{line-height:1.2}}.note{{background:#fff2d6;padding:16px;border-left:4px solid #d09a31}}table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dbe3e9;text-align:left}}th{{background:#eef3f7}}.chart{{max-width:1050px;width:100%}}details{{border:1px solid #d6e0e8;padding:12px;margin:14px 0;border-radius:7px}}summary{{cursor:pointer;font-weight:600}}.modalities{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}article{{background:#f4f7fa;padding:12px;border-radius:7px}}article img{{width:100%;display:block}}.caption{{white-space:pre-wrap;font-size:13px}}textarea{{width:100%;min-height:100px;box-sizing:border-box;margin-top:16px;padding:12px}}button{{padding:10px 16px;background:#177e89;color:white;border:0;border-radius:6px;cursor:pointer}}code{{word-break:break-all}}@media(max-width:750px){{.modalities{{grid-template-columns:1fr}}main{{padding:14px}}table{{font-size:11px}}}}</style></head><body><main><h1>Spatial social intelligence: expanded experiment</h1><p>Qwen3-VL-32B-Instruct, community 8-bit MLX checkpoint, frozen. 40 clips / 200 frames prepared. Stopped at user request: {s['saved_rows']}/640 condition rows saved. Scores below use the same {s['items']} fully evaluated clips in every condition; {s['excluded_incomplete_rows']} extra rows remain in the raw evidence.</p><div class='note'><strong>Measured computational results; author interpretation pending.</strong> No training was performed. Derived maps and generated captions can be wrong. This is a custom subset experiment, not an official full-benchmark result. TA approval and personal review have not been recorded.</div><h2>What changed</h2><p>RGB is compared with pose, COCO instance segmentation, relative monocular depth, and all three. Each is tested with images alone and with attached text: question-free RGB descriptions generated by the same 32B model, plus deterministic descriptions of the processed outputs. Repeated-image controls match the extra image slots; shuffled-all controls replace all auxiliary maps and their descriptions with another source. The captions are identical across conditions that use a given image. Text lengths are not matched. Pose extraction uses MediaPipe; segmentation uses YOLOv8n-seg; depth uses Depth Anything V2 Small. No correct labels, norm categories, or dataset-generated descriptions enter inference prompts.</p><img class='chart' src='comparison.png'><h2>{s['items']}-item matched results (partial experiment)</h2><table><tr><th rowspan='2'>Input</th><th colspan='3'>Images only</th><th colspan='3'>Images + descriptions</th></tr><tr><th>Action</th><th>Justification</th><th>Both</th><th>Action</th><th>Justification</th><th>Both</th></tr>{''.join(score_rows)}</table><details><summary>Paired differences and controls</summary><table><tr><th>Comparison</th><th>Joint difference and 95% CI</th><th>Improved</th><th>Worsened</th><th>Either prediction changed</th></tr>{''.join(changes)}</table><p>Intervals use 5,000 source-video bootstrap samples. A zero-width bootstrap interval from no observed paired changes is degenerate and is not proof of a zero population effect. All comparisons are exploratory; do not select a final model or tune prompts on these labels.</p></details>{agent_section}<p><a href="recoveries.html">Inspect the observed RGB-to-combined recovery cases and all their controls</a>.</p><h2>Inspect every image and its attached description</h2><p>Review notes save locally in this browser. Export them to keep a record.</p><button onclick='exportNotes()'>Export my review notes</button>{''.join(details)}<h2>Limits and disclosure</h2><ul><li>Segmentation is limited to COCO classes; black regions are not empty space. Masks/class predictions are unvalidated.</li><li>Depth is derived from RGB and has no calibrated distance scale or sensor ground truth. Its cross-frame scales can vary.</li><li>Pose is sparse and can miss partial bodies and hands. No ground-truth pose accuracy was measured.</li><li>RGB captions are generated by Qwen and may hallucinate or omit evidence. Caption effects do not establish a geometry mechanism.</li><li>The 32B run uses a new common prompt and should not be pooled with the earlier 8B pilot to claim a clean model-size effect.</li><li>The GPT agent pilot is a tool-using, context-accumulating workflow on five examples, not a statistically powered model comparison.</li><li>AI assistance includes design, code, model execution, captions, summaries, figures, and draft writing. Lead and revise the final interpretation yourself.</li></ul><p>Sources: <a href='https://huggingface.co/datasets/open-social-world/EgoNormia'>EgoNormia</a>, <a href='https://huggingface.co/mlx-community/Qwen3-VL-32B-Instruct-8bit'>Qwen checkpoint</a>, <a href='https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf'>Depth Anything V2</a>, <a href='https://docs.ultralytics.com/tasks/segment/'>Ultralytics segmentation</a>.</p></main><script>const prefix='egonormia-expanded-v1:';document.querySelectorAll('[data-key]').forEach(e=>{{e.value=localStorage.getItem(prefix+e.dataset.key)||'';e.addEventListener('input',()=>localStorage.setItem(prefix+e.dataset.key,e.value));}});function exportNotes(){{const responses={{}};document.querySelectorAll('[data-key]').forEach(e=>{{if(e.value)responses[e.dataset.key]=e.value;}});const blob=new Blob([JSON.stringify({{exported_at:new Date().toISOString(),responses}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='egonormia-expanded-review.json';a.click();URL.revokeObjectURL(a.href);}}</script></body></html>"""
    (OUT/'review.html').write_text(text)

def build_tex(s):
    tex=r"""% Draft section for the existing ICML report; author review required.
\section{Analysis: Explicit Geometry, Pose, and Descriptions}
\paragraph{Questions and protocol.} We investigate whether RGB-derived pose, instance segmentation, and relative depth improve zero-shot social-norm decisions, separately and together, and whether textual descriptions change the effect. We retain the fixed 40 source-disjoint EgoNormia examples and five pre-action frames per clip (200 frames). We use frozen MediaPipe body/hand extractors, YOLOv8n-seg COCO instance masks, and Depth Anything V2 Small relative depth. Maps are derived visual representations, not independent sensor modalities; estimated depth is not measured distance. A local community 8-bit Qwen3-VL-32B-Instruct checkpoint selects an action and then a justification with greedy decoding and independently permuted options, fixed across conditions. No training is performed.
\paragraph{Descriptions and controls.} Each representation is tested with images alone and with attached descriptions. RGB descriptions are generated by Qwen from the RGB grid without answer options, labels, taxonomy, or dataset descriptions. Processed-image descriptions deterministically summarize the extractor outputs and their limits. Repeated RGB controls match one or three auxiliary slots at identical RGB resolution. Shuffled-all uses another source's three maps and associated descriptions. Text lengths are not matched, so image-only/description ablations are essential. This custom pilot is not an official leaderboard replication.
\paragraph{Results.} The table reports joint action-and-justification correctness; full action/justification results and raw responses are in the repository.
\begin{center}\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrr}
Input & Images only & With descriptions \\
\hline
"""
    for b in BASES:
        tex+=LABELS[b].replace('RGB repeated four times',r'RGB $\times$ 4')+' & '+str(s['scores'][f'{b}_images']['both_correct']['correct'])+'/'+str(s['items'])+' & '+str(s['scores'][f'{b}_descriptions']['both_correct']['correct'])+'/'+str(s['items'])+' '+r'\\'+'\n'
    tex+=r'\end{tabular}\end{center}'+'\n'
    tex+=f"The user stopped inference after {s['saved_rows']} of 640 planned condition rows. All table entries use the same {s['items']} clips completed under all 16 conditions. The {s['excluded_incomplete_rows']} additional rows are preserved but excluded from this matched table. No further inference was performed.\n"
    for mode,label in [('images','images alone'),('descriptions','images and descriptions')]:
        d=next(d for d in s['paired_joint_differences'] if d['condition']==f'all_{mode}' and d['reference']==f'rgb_{mode}');m,lo,hi=d['joint_difference_ci']
        tex+=f"With {label}, all three minus RGB joint accuracy is {100*m:+.1f} percentage points; {d['improved']} items improve and {d['worsened']} worsen. "
        if not d['bootstrap_degenerate']:tex+=f"The paired 95\\% bootstrap interval is [{100*lo:+.1f}, {100*hi:+.1f}]. "
        else:tex+='No paired joint outcomes change; the empirical bootstrap interval is degenerate and cannot establish a zero population effect. '
    tex+='\n'
    tex+=f"RGB alone is jointly correct on {s['scores']['rgb_images']['both_correct']['correct']}/{s['items']}, compared with {s['scores']['all_images']['both_correct']['correct']}/{s['items']} for all-three images. Shuffled-all images reach {s['scores']['shuffled_all_images']['both_correct']['correct']}/{s['items']}, exceeding the aligned-map result. All-three with descriptions scores {s['scores']['all_descriptions']['both_correct']['correct']}/{s['items']}; therefore these observations do not establish a consistent advantage from aligned geometry or attached descriptions. The low absolute scores warrant review of the saved prompts and image evidence before interpreting the result as a model capability estimate. The matched cohort is the first fully processed prefix of the fixed ordering, not a subset selected for successful outcomes.\n"
    tex+=r"""\paragraph{Illustrative cases.} In the climbing-equipment example (\texttt{54adfe07}), RGB and pose-only select the incorrect none option, while segmentation-only, depth-only, and all-three images select the correct equipment-check action and justification. Repeated RGB still fails, but shuffled auxiliary maps also succeed, and descriptions remove the gain. Thus this recovery does not isolate an effect of aligned geometry. In the shopping example (\texttt{e93fdf27}), all-three images with descriptions select the correct friend-opinion action and justification, while RGB, individual representations, repeated RGB, and shuffled-all fail. Maps alone also fail there; unmatched description lengths and the combined visual/text change prevent a geometry-only interpretation. These examples should be inspected alongside the complete paired counts, rather than selected as evidence of general improvement.
"""
    if 'gpt56_failure_study' in s:
        study=s['gpt56_failure_study'];base=study['rgb_baseline'];n=study['diagnostic_n']
        tex+=f"\\paragraph{{Extended agent failure diagnostics.}} On the first 15 fixed-order RGB examples, the agent is jointly correct on {base['both_correct']}/{base['n']}. All {n} failures from the additional ten-case baseline were retested under the remaining 15 conditions. All-three images recover {study['conditions']['all_images']['both_correct']}/{n}; all-three with descriptions recover {study['conditions']['all_descriptions']['both_correct']}/{n}. The cohort is selected by baseline errors, and the agent accumulates context, so these are recovery diagnostics rather than unbiased accuracy or causal geometry estimates. Complete controls and image evidence are in the accompanying failure gallery.\n"
    if 'gpt56_agent_pilot' in s:
        tex+=r"\paragraph{GPT agent feasibility pilot.} A spawned agent configured with the requested \texttt{gpt-5.6-sol} override viewed actual image grids and selected answers without gold labels on the first five examples. Its context accumulates across cases and conditions, so this is an agent-workflow feasibility check, not a clean stateless API comparison or model ranking. "
        for c,v in s['gpt56_agent_pilot']['scores'].items():tex+=f"{c.replace('_',' ')}: {v['both_correct']}/{v['n']} joint-correct. "
        tex+='\n'
    tex+=r"""
\paragraph{Discussion (author review required).} Inspect which gains, declines, and answer changes survive the repeated-image and shuffled-map controls before attributing them to aligned geometry. Gains introduced only by descriptions may reflect language scaffolding or caption errors rather than new visual information. Sparse pose output, COCO-limited masks, and uncalibrated depth each require qualitative review. This small frozen-model experiment does not test the proposal's fine-tuning, and the 32B run's new prompt prevents a clean size-only comparison with the earlier 8B pilot. EgoNormia labels were used for exploratory scoring, so it remains excluded from training but is not untouched. Preserve failed outputs and evaluate any future adaptation on independently reserved examples under matched budgets.
% AI DISCLOSURE: Codex designed and implemented this expanded experiment, generated summaries/figures and drafted this section. Qwen generated RGB captions and experimental predictions; MediaPipe, YOLO, and Depth Anything generated derived representations. A GPT-5.6-sol-configured agent made a blinded tool-using pilot. Human interpretation and review remain pending.
"""
    (OUT/'idea5-expanded-section.tex').write_text(tex)

if __name__=='__main__':main()
