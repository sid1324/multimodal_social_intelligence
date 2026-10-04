"""Summarize measured outputs and build a local review report and LaTeX section."""
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"output/idea5-audit"

def interval(values,seed=42):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=x[rng.integers(0,len(x),size=(5000,len(x)))].mean(axis=1)
    return [float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]

def exact_mcnemar(up,down):
    n=up+down
    return min(1.,2*sum(math.comb(n,k) for k in range(min(up,down)+1))/2**n) if n else 1.

def main():
    protocol=json.loads((OUT/"protocol.json").read_text());frames=json.loads((OUT/"frame_results.json").read_text())
    samples=protocol["sample"];ids=[x["id"] for x in samples]
    assert len(ids)==len(set(x["source"] for x in samples))
    assert len(frames)==len(ids)*5
    assert all(len([r for r in frames if r['id']==i])==5 for i in ids)
    metrics={"body_50":lambda r:r["body_available_50"],"body_80":lambda r:r["body_available_80"],"hand":lambda r:r["raw_hands"]>0,"either":lambda r:r["body_available_50"] or r["raw_hands"]>0,"both":lambda r:r["body_available_50"] and r["raw_hands"]>0}
    clip_values={m:[sum(fn(r) for r in frames if r["id"]==i)/5 for i in ids] for m,fn in metrics.items()}
    coverage={m:{"frames":sum(fn(r) for r in frames),"frame_fraction_cluster_ci":interval(clip_values[m]),"clips_with_at_least_one":sum(v>0 for v in clip_values[m]),"clips_all_five":sum(v==1 for v in clip_values[m])} for m,fn in metrics.items()}
    categories=sorted(set(c for x in samples for c in x["categories"]))
    by_category=[]
    for c in categories:
        chosen=[r for r in frames if c in r["categories"]]; clip_ids=[x["id"] for x in samples if c in x["categories"]]
        by_category.append({"category":c,"clips":len(clip_ids),"frames":len(chosen),"body_50_fraction":sum(r["body_available_50"] for r in chosen)/len(chosen),"hand_fraction":sum(r["raw_hands"]>0 for r in chosen)/len(chosen)})
    summary={"clips":len(ids),"frames":len(frames),"coverage":coverage,"by_category":by_category,"raw_pose_frames":sum(r["raw_bodies"]>0 for r in frames),"all_raw_bodies":sum(r["raw_bodies"] for r in frames),"all_raw_hands":sum(r["raw_hands"] for r in frames),"inference_seconds":sum(r["inference_seconds"] for r in frames),"human_annotation":"Not performed; detector-output availability, not accuracy or recall"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(7.4,4.3)); keys=["body_50","body_80","hand","either"]
    means=np.array([coverage[k]["frame_fraction_cluster_ci"][0] for k in keys])*100
    lows=np.array([coverage[k]["frame_fraction_cluster_ci"][1] for k in keys])*100
    highs=np.array([coverage[k]["frame_fraction_cluster_ci"][2] for k in keys])*100
    ax.bar(range(4),means,color=["#177e89","#75b5bc","#8c5bb6","#334e68"],width=.65)
    ax.errorbar(range(4),means,yerr=[means-lows,highs-means],fmt="none",ecolor="#203040",capsize=4)
    for j,k in enumerate(keys):ax.text(j,highs[j]+2,f"{coverage[k]['frames']}/{len(frames)}",ha="center",fontsize=10)
    ax.set_xticks(range(4),["Body, threshold .5","Body, threshold .8","Hand output","Either (.5 body)"]); ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel("Sampled frames with output (%)")
    ax.set_title(f"Pose representation availability: {len(ids)} clips, {len(frames)} frames",loc="left",fontweight="bold",pad=16)
    fig.text(.02,.015,"95% bootstrap intervals resample source clips. Output availability is not detection accuracy.",fontsize=9,color="#485767")
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(OUT/"coverage.png",dpi=220);plt.close(fig)
    qwen=[]; qwen_path=OUT/"qwen_results.json"
    if qwen_path.exists():
        qwen=json.loads(qwen_path.read_text())
        conditions=json.loads((OUT/"qwen_protocol.json").read_text())["conditions"]
        assert len(qwen)==len(ids)*len(conditions),"VLM comparison is incomplete"
        lookup={(r["id"],r["condition"]):r for r in qwen}
        scores={c:{m:{"correct":sum(lookup[(i,c)][m] for i in ids),"n":len(ids),"ci":interval([lookup[(i,c)][m] for i in ids])} for m in ["action_correct","justification_correct","both_correct"]} for c in conditions}
        invalid={c:sum(lookup[(i,c)]["action_original_idx"] is None or lookup[(i,c)]["justification_original_idx"] is None for i in ids) for c in conditions}
        comparisons=[]
        for a,b in [("rgb_pose","rgb"),("rgb_pose","rgb_repeat"),("rgb_pose","rgb_shuffled_pose")]:
            differences=[int(lookup[(i,a)]["both_correct"])-int(lookup[(i,b)]["both_correct"]) for i in ids]; up=differences.count(1);down=differences.count(-1)
            comparisons.append({"condition":a,"reference":b,"metric":"both_correct","difference_ci":interval(differences),"improved":up,"worsened":down,"unchanged":len(ids)-up-down,"action_answer_flips":sum(lookup[(i,a)]['action_original_idx']!=lookup[(i,b)]['action_original_idx'] for i in ids),"justification_answer_flips":sum(lookup[(i,a)]['justification_original_idx']!=lookup[(i,b)]['justification_original_idx'] for i in ids),"exact_mcnemar_p":exact_mcnemar(up,down)})
        summary["qwen"]={"scores":scores,"invalid_outputs":invalid,"comparisons":comparisons,"complete":True}
        fig,ax=plt.subplots(figsize=(7.4,4.3)); x=np.arange(len(conditions)); width=.24
        for j,(metric,label,color) in enumerate([("action_correct","Action","#177e89"),("justification_correct","Justification","#8c5bb6"),("both_correct","Both","#334e68")]):
            vals=[scores[c][metric]["correct"]/len(ids)*100 for c in conditions]
            bars=ax.bar(x+(j-1)*width,vals,width,label=label,color=color)
            ax.bar_label(bars,labels=[str(scores[c][metric]["correct"])+f"/{len(ids)}" for c in conditions],padding=3,fontsize=8)
        ax.set_xticks(x,["RGB","RGB + repeated RGB","RGB + aligned pose","RGB + shuffled pose"],fontsize=9);ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel("Correct responses (%)");ax.legend(frameon=False,ncol=3,loc="upper right")
        ax.set_title("Frozen Qwen3-VL-8B: fixed subset comparison",loc="left",fontweight="bold",pad=16)
        fig.text(.02,.015,"4-bit local model; custom prompts; no training. Paired intervals and controls are in the report.",fontsize=9,color="#485767")
        fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(OUT/"qwen_comparison.png",dpi=220);plt.close(fig)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    build_html(protocol,frames,summary,qwen)
    build_tex(protocol,summary)
    print(json.dumps(summary,indent=2))

def build_html(protocol,frames,summary,qwen):
    esc=html.escape; ids=[x["id"] for x in protocol["sample"]]
    coverage_rows="".join(f"<tr><td>{esc(k)}</td><td>{v['frames']}/{summary['frames']}</td><td>{v['clips_with_at_least_one']}/{summary['clips']}</td><td>{100*v['frame_fraction_cluster_ci'][0]:.1f}% [{100*v['frame_fraction_cluster_ci'][1]:.1f}, {100*v['frame_fraction_cluster_ci'][2]:.1f}]</td></tr>" for k,v in summary["coverage"].items())
    cat_rows="".join(f"<tr><td>{esc(r['category'])}</td><td>{r['clips']}</td><td>{100*r['body_50_fraction']:.1f}%</td><td>{100*r['hand_fraction']:.1f}%</td></tr>" for r in summary["by_category"])
    qwen_section=""
    if qwen:
        stats=summary["qwen"]
        rows="".join(f"<tr><td>{esc(c)}</td>"+"".join(f"<td>{stats['scores'][c][m]['correct']}/{summary['clips']}</td>" for m in ['action_correct','justification_correct','both_correct'])+f"<td>{stats['invalid_outputs'][c]}</td></tr>" for c in stats['scores'])
        diffs="".join(f"<tr><td>Pose vs {esc(r['reference'])}</td><td>"+("0.0 pp; no observed paired changes" if r['improved']+r['worsened']==0 else f"{100*r['difference_ci'][0]:+.1f} pp [{100*r['difference_ci'][1]:+.1f}, {100*r['difference_ci'][2]:+.1f}]")+f"</td><td>{r['improved']}</td><td>{r['worsened']}</td><td>{r['exact_mcnemar_p']:.4f}</td></tr>" for r in stats['comparisons'])
        qwen_section=f"<h2>Frozen-model comparison</h2><p>Qwen3-VL-8B-Instruct community 4-bit model, greedy decoding, no fine-tuning. Two sequential prompts select action and justification; options are independently permuted but kept identical across conditions. Invalid outputs count as wrong. This custom subset pilot does not replicate official leaderboard prompting.</p><img class='chart' src='qwen_comparison.png'><table><tr><th>Input</th><th>Action</th><th>Justification</th><th>Both</th><th>Invalid</th></tr>{rows}</table><table><tr><th>Paired joint comparison</th><th>Difference, 95% bootstrap CI</th><th>Improved</th><th>Worsened</th><th>McNemar p</th></tr>{diffs}</table><p>Intervals resample source videos (one item per source). Tests are exploratory and unadjusted. A small or null gain cannot disprove the fine-tuning proposal; this model was not trained to interpret these rendered maps.</p>"
    sections=[]
    for index,item in enumerate(protocol['sample']):
        ff=sorted([r for r in frames if r['id']==item['id']],key=lambda r:r['sample_index'])
        cards=[]
        for r in ff:
            key=f"{item['id']}:{r['sample_index']}"
            cards.append(f"<article><h4>Frame {r['sample_index']+1}: {r['timestamp_seconds']:.2f}s</h4><img loading='lazy' src='{r['paths']['rgb']}' alt='Original frame'><img loading='lazy' src='{r['paths']['overlay']}' alt='Body and hand overlay'><img loading='lazy' src='{r['paths']['skeleton']}' alt='Skeleton-only map'><p>Raw bodies: {r['raw_bodies']}; hands: {r['raw_hands']}; retained landmarks: {r['body_retained_50']}</p><label>Body overlay review<select data-key='{key}:body'><option value=''>Unreviewed</option><option>No relevant body visible</option><option>Mostly correct</option><option>Partly correct</option><option>Incorrect</option><option>Uncertain</option></select></label><label>Hand overlay review<select data-key='{key}:hand'><option value=''>Unreviewed</option><option>No relevant hand visible</option><option>Mostly correct</option><option>Partly correct</option><option>Incorrect</option><option>Uncertain</option></select></label><textarea data-key='{key}:notes' placeholder='Which people or hands are missed? Is the detected pose useful for the action?'></textarea></article>")
        ann=item['annotation']; action_list=''.join(f"<li>{esc(a) if a else '[None]'}</li>" for a in ann['behaviors'])
        qrows=''.join(f"<tr><td>{esc(r['condition'])}</td><td>{r['action_original_idx']+1 if r['action_original_idx'] is not None else 'Invalid'}</td><td>{r['justification_original_idx']+1 if r['justification_original_idx'] is not None else 'Invalid'}</td><td>{'Yes' if r['both_correct'] else 'No'}</td></tr>" for r in qwen if r['id']==item['id'])
        result_table=f"<table><tr><th>Input</th><th>Action</th><th>Justification</th><th>Both correct</th></tr>{qrows}</table>" if qrows else ''
        sections.append(f"<details class='clip'><summary>{index+1}. {esc(item['id'])} - {esc(', '.join(item['categories']))}</summary><ol>{action_list}</ol><details><summary>Show gold annotation</summary><p>Action {ann['correct']+1}: {esc(ann['behaviors'][ann['correct']])}<br>Justification: {esc(ann['justifications'][ann['correct']])}</p></details>{result_table}<label>Spatial evidence needed?<select data-key='{item['id']}:spatial'><option value=''>Unreviewed</option><option>Distance or relative position</option><option>Body orientation</option><option>Hands or object interaction</option><option>Multiple spatial cues</option><option>No clear spatial dependency</option><option>Uncertain</option></select></label><div class='frames'>{''.join(cards)}</div></details>")
    text=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>EgoNormia - Idea 5 audit</title><style>body{{font:16px/1.55 system-ui,sans-serif;background:#f4f7fa;color:#203040;margin:0}}main{{max-width:1200px;margin:32px auto;padding:24px;background:white;border-radius:12px}}h1,h2{{line-height:1.2}}.note{{padding:16px;background:#fff2d4;border-left:4px solid #dba23b}}table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}th{{background:#edf3f7}}.chart{{max-width:760px;width:100%}}summary{{cursor:pointer;font-weight:600;padding:12px}}details.clip{{border:1px solid #d6e0e8;margin:14px 0;border-radius:8px;padding:10px}}.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-top:20px}}article{{background:#f4f7fa;padding:10px;border-radius:8px}}article img{{width:100%;display:block;margin-bottom:6px}}article p{{font-size:12px}}label{{display:block;margin:8px 0}}select,textarea{{width:100%;padding:8px;border:1px solid #bac7d2;border-radius:4px;box-sizing:border-box}}textarea{{min-height:90px}}button{{padding:12px 18px;background:#177e89;color:white;border:0;border-radius:6px;cursor:pointer}}code{{word-break:break-all}}footer{{font-size:13px;color:#526475;margin-top:32px}}</style></head><body><main><h1>EgoNormia: geometry and pose feasibility</h1><p>Idea 5 exploratory analysis - {summary['clips']} independent source clips, {summary['frames']} pre-action frames.</p><div class='note'><strong>Draft for author review.</strong> Human review and TA approval have not been recorded. Automatic landmark availability is not detection correctness, recall, or evidence that social norms require pose. Inspect the gallery and write your own interpretation before submitting. AI assistance must be disclosed.</div><h2>What was measured</h2><p>Sources and examples were selected by a fixed SHA256 rule before inference. Five frames were decoded at 10%, 30%, 50%, 70%, and 90% of each pre-action clip. Frozen MediaPipe Full body and Hand models ran in independent IMAGE mode, up to four detections per frame, detection and presence thresholds 0.5. Body output is retained when at least eight of 33 landmarks are inside the image and have both visibility and presence at least 0.5 or 0.8. This cutoff is an operational rule, not a validated quality metric. Hand outputs have no equivalent per-keypoint scores. No temporal tracking, training, depth estimation, or segmentation was performed.</p><img class='chart' src='coverage.png'><table><tr><th>Output rule</th><th>Frames</th><th>Clips with any output</th><th>Frame rate and 95% clustered CI</th></tr>{coverage_rows}</table><h2>Descriptive category breakdown</h2><p>Categories refer to the annotated correct action, overlap, and have small unequal sample counts. Detection availability does not establish which norm needs a representation.</p><table><tr><th>Norm category</th><th>Clips</th><th>Body output (.5)</th><th>Hand output</th></tr>{cat_rows}</table>{qwen_section}<h2>Review the original frames and outputs</h2><p>Each frame shows original RGB, overlay, and skeleton-only image. Yellow indicates body points retained at .5; magenta indicates predicted hand points. Blank maps remain in the sample. Record mistaken and missed detections; the wearer may show only hands, so absence of a full-body skeleton is not necessarily an error.</p><p>Your review is saved in this browser locally. Export it to preserve a review record; it is not sent anywhere.</p><button onclick='exportReview()'>Export my review as JSON</button>{''.join(sections)}<h2>Limitations and next decisions</h2><ul><li>The 40-source sample is exploratory; it is not a full benchmark evaluation or a representative indoor subset.</li><li>No ground-truth pose or hand annotations are available in this audit. Do not report precision, recall, or detection accuracy without human labels.</li><li>Five independent frames do not measure motion or identity continuity; detector caps can miss crowded interactions.</li><li>Derived skeleton maps are visual representations of RGB; they are not independent sensor modalities.</li><li>Inspect task-relevant missing hands, partial bodies, occlusion, and spatial cues before deciding which representation to retain.</li><li>EgoNormia labels were inspected for scoring and descriptions were not passed to the VLM. Future training/model-selection claims must acknowledge this exploratory exposure.</li></ul><h2>Reproduce</h2><p>From the project folder:<br><code>.venv/bin/python scripts/download_egonormia.py</code><br><code>.venv-pose/bin/python scripts/audit_egonormia_pose.py</code><br><code>.venv/bin/python scripts/compare_qwen_pose.py</code><br><code>.venv-pose/bin/python scripts/build_audit_report.py</code></p><footer>Sources: <a href='https://huggingface.co/datasets/open-social-world/EgoNormia'>EgoNormia release</a>; <a href='https://github.com/Open-Social-World/EgoNormia'>official evaluation annotations</a>; <a href='https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python'>MediaPipe Pose</a>; <a href='https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python'>MediaPipe Hand</a>. Protocol, results, model hashes, package versions, and source revisions are saved alongside this report.</footer></main><script>const prefix='egonormia-pose-audit-v1:';document.querySelectorAll('[data-key]').forEach(e=>{{e.value=localStorage.getItem(prefix+e.dataset.key)||'';e.addEventListener('input',()=>localStorage.setItem(prefix+e.dataset.key,e.value));}});function exportReview(){{const responses={{}};document.querySelectorAll('[data-key]').forEach(e=>{{if(e.value)responses[e.dataset.key]=e.value;}});const blob=new Blob([JSON.stringify({{audit:'v1',exported_at:new Date().toISOString(),responses}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='egonormia-human-review.json';a.click();URL.revokeObjectURL(a.href);}}</script></body></html>"""
    (OUT/'review.html').write_text(text)

def build_tex(protocol,s):
    c=s['coverage']; n=s['frames']; clips=s['clips']
    tex=r"""% Standalone section for the team's existing ICML file.
% Requires graphicx and the bibliography entries in idea5-references.bib.
% Replace provisional discussion after personal review. Do not claim TA approval or human annotation.
\section{Analysis: Availability of Body and Hand Pose in EgoNormia}
\paragraph{Motivation and questions.} Idea 5 proposes adding explicit geometry and body/hand representations to RGB inputs for social-norm reasoning. We examine two premises: whether pretrained extractors produce candidate body/hand representations often enough to supply this pathway, and whether aligned maps provide an immediate benefit to a frozen VLM beyond extra-image and mismatched-map controls. We use EgoNormia's pre-action visual context \citep{rezaei2025egonormia}. This is an output-availability audit and a zero-shot pilot, not a test of pose accuracy or the proposed fine-tuning.
\paragraph{Data and protocol.} We select CLIPS items from distinct source videos using a fixed SHA256 ordering before detector inference, choosing one item per source. Five frames per pre-action clip are sampled at 10\%, 30\%, 50\%, 70\%, and 90\% of its length (FRAMES frames total). We apply frozen MediaPipe Pose Landmarker Full and Hand Landmarker models in independent-image mode, allowing up to four bodies and four hands. Detection and presence thresholds are 0.5. A body output is retained if at least eight of its 33 landmarks lie within the image and have predicted visibility and presence at least 0.5; we repeat the retention rule at 0.8 as a sensitivity check. The eight-point cutoff is an operational filter, not a validated measure of quality. Hand output means at least one predicted hand; the model does not supply equivalent per-landmark scores. We retain all examples, including blank outputs. Intervals use 5,000 source-video bootstrap samples.
\paragraph{Results.} Body output satisfies the 0.5 retention rule in BODY50/FRAMES sampled frames (BODY50P\%) and the 0.8 rule in BODY80/FRAMES (BODY80P\%). At least one hand is predicted in HAND/FRAMES frames (HANDP\%). At least one body or hand output is available in EITHER/FRAMES frames (EITHERP\%); CLIPANY/CLIPS clips contain such an output in at least one sampled frame. These are detector-output frequencies, not precision, recall, or human-validated extraction quality.
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{coverage.png}
\caption{Availability of body and hand outputs on the fixed sample. Error bars are 95\% bootstrap intervals clustered by source video. Body outputs require at least eight retained landmarks; predicted hand outputs use the detector's standard acceptance threshold.}
\end{figure}
"""
    replacements={'CLIPS':str(clips),'FRAMES':str(n),'BODY50P':f"{100*c['body_50']['frames']/n:.1f}",'BODY80P':f"{100*c['body_80']['frames']/n:.1f}",'HANDP':f"{100*c['hand']['frames']/n:.1f}",'EITHERP':f"{100*c['either']['frames']/n:.1f}",'BODY50':str(c['body_50']['frames']),'BODY80':str(c['body_80']['frames']),'HAND':str(c['hand']['frames']),'EITHER':str(c['either']['frames']),'CLIPANY':str(c['either']['clips_with_at_least_one'])}
    for key in sorted(replacements,key=len,reverse=True):tex=tex.replace(key,replacements[key])
    if 'qwen' in s:
        q=s['qwen'];scores=q['scores'];comparison=q['comparisons'][0]
        tex+=r"""
\paragraph{Frozen-model pilot.} On the same fixed items, we compare a local community 4-bit Qwen3-VL-8B-Instruct checkpoint under RGB-only, RGB plus repeated RGB, RGB plus aligned body/hand skeletons, and RGB plus skeletons from a different source video. RGB grids contain the same five chronological frames, each resized to a maximum side of 384 pixels; auxiliary grids match target grid dimensions. Repeated RGB controls for the extra image slot, and cyclically shuffled maps test alignment. Greedy decoding separately selects an action and then its justification. Action and justification options are independently permuted per item and kept fixed across conditions; no gold labels, norm labels, or generated descriptions enter prompts. Invalid outputs count as wrong. This is a custom subset pilot, not an official leaderboard replication.
\begin{center}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}
Input & Action & Justification & Both \\
\hline
"""
        for label,condition in [('RGB','rgb'),('RGB + repeat','rgb_repeat'),('RGB + aligned pose','rgb_pose'),('RGB + shuffled pose','rgb_shuffled_pose')]:
            tex+=label+' & '+' & '.join(f"{scores[condition][m]['correct']}/{clips}" for m in ['action_correct','justification_correct','both_correct'])+r" \\"+'\n'
        tex+=r"\end{tabular}\end{center}"+'\n'
        mean,lo,hi=comparison['difference_ci']
        tex+=f"Aligned pose minus RGB joint accuracy is {100*mean:+.1f} percentage points (95\\% paired bootstrap interval [{100*lo:+.1f}, {100*hi:+.1f}]); {comparison['improved']} items improve and {comparison['worsened']} worsen. "
        for comp in q['comparisons'][1:]:
            mean,lo,hi=comp['difference_ci']; label='repeated RGB' if comp['reference']=='rgb_repeat' else 'shuffled pose'
            if comp['improved']+comp['worsened']==0:
                tex+=f"Against {label}, no item's joint correctness changes. The resulting zero-width empirical bootstrap interval is degenerate and does not establish a zero population effect. "
            else:
                tex+=f"Against {label}, the joint difference is {100*mean:+.1f} points [{100*lo:+.1f}, {100*hi:+.1f}]. "
        tex+='\n'
    if 'qwen' in s:
        tex+=r"""
\paragraph{Discussion of this analysis (provisional; author review required).} Aligned pose produces no observed joint-accuracy gain over RGB on this sample, and aligned and shuffled pose have identical joint-correctness outcomes on all 40 items. This provides no evidence of an alignment-dependent benefit under the tested setup. Repeating RGB also changes outcomes, demonstrating why an extra-image control matters. These results do not disprove the fine-tuning proposal: the frozen model has not been adapted to the rendered maps, and the sample is small. Candidate body or hand outputs are available in only 40.5\% of sampled frames; human review must distinguish absent people/hands from detector failures or task-irrelevant detections. Egocentric views can show the wearer's hands without a full body, so both extractors need separate assessment. Five independent frames do not measure motion or identity continuity. Before committing to pose augmentation, manually audit overlays, preserve missing-output indicators, and evaluate adaptation under matched budgets. EgoNormia is excluded from training but has been exposed during exploratory analysis; it should not be described as untouched.
"""
    else:
        tex+=r"""
\paragraph{Discussion of this analysis (provisional; author review required).} These output frequencies describe candidate representation availability rather than correctness or task relevance. Human review must separate absent people/hands from detector failures. Egocentric views can show hands without a full body. Preserve missing-output indicators and verify task relevance before committing to pose augmentation.
"""
    tex+=r"""
% AUTHOR TASK: Replace the provisional discussion with your observations after reviewing overlays.
% AI DISCLOSURE: An OpenAI Codex agent selected the reproducible sampling rule, wrote and debugged code, downloaded data/models, ran extraction/inference, generated figures and numerical summaries, and drafted this section. MediaPipe generated body/hand landmarks. Qwen generated experimental action/justification predictions. Human review/interpretation is pending and must be attributed truthfully after completion.
"""
    (OUT/'idea5-section.tex').write_text(tex)
    (OUT/'idea5-references.bib').write_text(r"""@inproceedings{rezaei2025egonormia,
  title={EgoNormia: Benchmarking Physical-Social Norm Understanding},
  author={Rezaei, MohammadHossein and Fu, Yicheng and Cuvin, Phil and Ziems, Caleb and Zhang, Yanzhe and Zhu, Hao and Yang, Diyi},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}, doi={10.18653/v1/2025.findings-acl.985},
  url={https://aclanthology.org/2025.findings-acl.985/}}
""")

if __name__=='__main__':main()
