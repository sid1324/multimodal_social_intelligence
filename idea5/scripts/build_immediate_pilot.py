"""Snapshot the first three complete cases without stopping the extensive run."""
import copy
import re
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_qwen_pose import permutation
import build_expanded_report as report

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'output/idea5-expanded';OUT=SOURCE/'pilot';OUT.mkdir(exist_ok=True)
inputs=json.loads((SOURCE/'inputs.json').read_text());rows=json.loads((SOURCE/'qwen_results.json').read_text());conditions=[f'{b}_{m}' for m in ['images','descriptions'] for b in report.BASES]
selected=inputs['items'][:3];ids=[x['id'] for x in selected];rows=[x for x in rows if x['id'] in ids];lookup={(x['id'],x['condition']):x for x in rows};assert len(rows)==48 and all((i,c) in lookup for i in ids for c in conditions)
summary={'snapshot':'First three fixed-order source clips with all 16 conditions complete; no outcome-based filtering','n':3,'frames':15,'model':json.loads((SOURCE/'model.json').read_text()),'scores':{},'paired_joint_differences':[]}
for c in conditions:
 summary['scores'][c]={metric:{'correct':sum(lookup[(i,c)][metric] for i in ids),'n':3,'ci':report.bootstrap([lookup[(i,c)][metric] for i in ids])} for metric in ['action_correct','justification_correct','both_correct']}
for mode in ['images','descriptions']:
 for base in ['pose','segmentation','depth','all']:
  a=f'{base}_{mode}';b=f'rgb_{mode}';values=[int(lookup[(i,a)]['both_correct'])-int(lookup[(i,b)]['both_correct']) for i in ids]
  summary['paired_joint_differences'].append({'condition':a,'reference':b,'joint_difference_ci':report.bootstrap(values),'improved':values.count(1),'worsened':values.count(-1),'bootstrap_degenerate':not any(values),'either_prediction_changed':sum(any(lookup[(i,a)][k]!=lookup[(i,b)][k] for k in ['action_original_idx','justification_original_idx']) for i in ids)})
private={x['id']:x['annotation'] for x in json.loads((ROOT/'output/idea5-audit/protocol.json').read_text())['sample']}
agent=json.loads((SOURCE/'gpt56-agent-results.json').read_text());agent_scores={};agent_scored=[]
for x in agent['items']:
 if x['id'] not in ids:continue
 a=permutation(x['id'],'action')[x['action_position']-1];j=permutation(x['id'],'justification')[x['justification_position']-1];gold=private[x['id']]['correct'];c='rgb_images' if x['condition']=='rgb' else x['condition'];agent_scored.append({'id':x['id'],'condition':c,'action_correct':a==gold,'justification_correct':j==gold,'both_correct':a==gold and j==gold})
for c in sorted({x['condition'] for x in agent_scored}):
 rr=[x for x in agent_scored if x['condition']==c];agent_scores[c]={'n':len(rr),'action_correct':sum(x['action_correct'] for x in rr),'justification_correct':sum(x['justification_correct'] for x in rr),'both_correct':sum(x['both_correct'] for x in rr),'qwen_matched_both':sum(lookup[(i,c)]['both_correct'] for i in ids),'qwen_matched_n':3}
summary['gpt56_agent_pilot']={'scores':agent_scores,'identity':'Agent configured with requested gpt-5.6-sol override; accepted by orchestration but not independently inspected inside agent runtime.','protocol_limit':'These three cases are matched with Qwen. Agent uses tools and accumulated context, so this is not a clean model ranking. The full agent pilot has five cases; this immediate report subsets it to the same three.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'qwen_results.json').write_text(json.dumps(rows,indent=2));(OUT/'gpt56-agent-scored.json').write_text(json.dumps(agent_scored,indent=2))
html_inputs=copy.deepcopy(inputs);html_inputs['items']=copy.deepcopy(selected)
for item in html_inputs['items']:
 item['images']={k:'../'+v for k,v in item['images'].items()}
report.OUT=OUT;report.build_html(html_inputs,summary,lookup)
html=(OUT/'review.html').read_text().replace('Same 40 source clips / 200 frames.','First three completed source clips / 15 frames. The full 40-clip run continues.').replace('40-item results','Immediate 3-item results').replace('The agent viewed actual image grids and was not supplied gold labels.','The agent viewed actual image grids and was not supplied gold labels.').replace('only the same five examples','only the same three examples').replace('The computational work','The pilot computational work')
html=re.sub(r'(?<=\d)/40\b','/3',html)
html=html.replace('<h1>Spatial social intelligence: expanded experiment</h1>','<h1>Immediate three-example pilot</h1><p><strong>Preliminary snapshot:</strong> three examples cannot establish a general improvement or model ranking. The extensive report will appear in the parent folder after the run completes.</p>')
(OUT/'review.html').write_text(html)
fig,ax=plt.subplots(figsize=(7,3));bases=report.BASES[:5];x=list(range(5));ax.bar([v-.18 for v in x],[summary['scores'][b+'_images']['both_correct']['correct'] for b in bases],width=.36,label='Images only',color='#177e89');ax.bar([v+.18 for v in x],[summary['scores'][b+'_descriptions']['both_correct']['correct'] for b in bases],width=.36,label='Images + descriptions',color='#8c5bb6');ax.set_xticks(x,['RGB','+ Pose','+ Segmentation','+ Depth','+ All']);ax.set_yticks([0,1,2,3]);ax.set_ylim(0,3.5);ax.set_ylabel('Both correct (out of 3)');ax.set_title('Immediate Qwen3-VL-32B pilot: no answer changes observed');ax.legend(frameon=False,ncol=2,loc='upper center');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
tex=r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.6in]{geometry}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\begin{document}
\fontsize{9.5}{11}\selectfont
\begin{center}{\Large\bfseries EgoNormia: Immediate Three-Example Pilot}\\[5pt]October 4, 2026\end{center}
\noindent\textbf{Status.} This is an immediate snapshot of the first three fixed-order clips whose 16 conditions are complete. The extensive 40-clip run continues separately. These three examples were not chosen by performance. No training was performed.
\paragraph{Setup.} Qwen3-VL-32B-Instruct, community 8-bit MLX checkpoint, runs locally on an M5 Max with 128 GiB RAM. Each clip has five pre-action frames. Inputs are RGB alone, RGB plus pose, RGB plus COCO instance segmentation, RGB plus relative depth, or RGB plus all three. Each is tested with images alone and with attached text. RGB captions are generated from RGB without questions, answer options, or labels; processed descriptions summarize extractor outputs. Repeated-RGB and shuffled-map controls are also tested. Actions are selected first; justifications are conditioned on the selected action. Options are permuted identically across conditions.
\begin{center}\small
\begin{tabular}{lccc}
 & Action correct & Justification correct & Both correct\\\hline
'''
for b in report.BASES:
 for mode,label in [('images','images'),('descriptions','images + text')]:
  scores=summary['scores'][b+'_'+mode];name=report.LABELS[b].replace('RGB repeated four times',r'RGB $\times$ 4')
  tex+=name+' ('+label+') & '+' & '.join(str(scores[m]['correct'])+'/3' for m in ['action_correct','justification_correct','both_correct'])+r'\\'+'\n'
tex+=r'''\end{tabular}\end{center}
\paragraph{Observed result.} Qwen gets the kitchen/tray case correct and selects the none option for both action and justification in the construction and automotive/cloth cases, where that option is incorrect. Its selected action and justification are unchanged across all 16 conditions in all three examples. Thus every condition has 1/3 joint correctness; there are zero gains, zero declines, and zero answer changes in this snapshot. This does not establish that geometry or descriptions have no effect on the larger dataset or after training.
\paragraph{Requested GPT agent pilot.} A spawned agent configured with the requested \texttt{gpt-5.6-sol} override viewed the actual image grids without gold labels. On the same three cases, joint correctness is:
\begin{center}\small\begin{tabular}{lr}
Condition & Agent both correct\\\hline
'''
for c,v in agent_scores.items():tex+=c.replace('_',' ')+' & '+str(v['both_correct'])+'/3 '+r'\\'+'\n'
tex+=r'''\end{tabular}\end{center}
\noindent The agent's answers also do not change across its conditions. Its context accumulates and it uses image-viewing tools, unlike Qwen's stateless local calls. The requested override was accepted by orchestration but not independently inspected inside the agent. These are workflow observations, not a controlled model ranking. The full agent pilot has five cases; this report shows the matched three.
\paragraph{Representation limits and review.} Body/hand pose comes from frozen MediaPipe models, segmentation from YOLOv8n-seg, and depth from Depth Anything V2 Small. All are derived from RGB. Depth has no measured distance scale; segmentation misses non-COCO objects and can make class errors (including horse/bird predictions in the construction sequence). Captions can be wrong. Review the attached RGB, pose, segmentation, and depth images and descriptions in the companion gallery; it also shows every selected action and justification. With only three cases, no population effect or statistical significance is claimed.
\paragraph{AI disclosure.} Codex designed and implemented the computation, generated summary statistics and figures, and drafted this report. Qwen generated RGB captions and predictions; pretrained extractors generated maps; a GPT-configured agent generated blinded pilot predictions. Student interpretation, personal review, and TA approval remain pending. This is a draft for review, not the complete team assignment.
\end{document}
'''
(OUT/'immediate-report.tex').write_text(tex)
print(json.dumps({'qwen':summary['scores']['rgb_images'],'agent':agent_scores,'snapshot_rows':len(rows)},indent=2))
