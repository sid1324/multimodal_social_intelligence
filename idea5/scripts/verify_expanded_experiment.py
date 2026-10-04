"""Check saved experimental evidence and deliverable links after the complete run."""
import hashlib
import json
import re
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image
from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded';OLD=ROOT/'output/idea5-audit'

class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value and not value.startswith(('http:','https:','#','data:')):self.paths.append(value)

def main():
    inputs=json.loads((OUT/'inputs.json').read_text())['items'];assert len(inputs)==40 and len({x['source'] for x in inputs})==40
    geometries=json.loads((OUT/'geometry_results.json').read_text());assert len(geometries)==200
    raw_shape_counts=Counter();decoded=0
    for row in geometries:
        raw=np.load(OUT/row['depth_raw_path']);assert raw.ndim==2 and np.isfinite(raw).all();raw_shape_counts[str(raw.shape)]+=1
        for role in ['segmentation_path','depth_path']:
            with Image.open(OUT/row[role]) as image:image.load();assert image.size==(row['width'],row['height']);decoded+=1
    for item in inputs:
        assert set(item['images'])==set(item['descriptions'])=={'rgb','pose','segmentation','depth'}
        sizes=[]
        for path in item['images'].values():
            with Image.open(OUT/path) as image:image.load();sizes.append(image.size)
        assert len(set(sizes))==1 and all(item['descriptions'].values())
    ann={x['id']:x['annotation'] for x in json.loads((OLD/'protocol.json').read_text())['sample']}
    manifest={x['id']:x for x in json.loads((OUT/'full-blind-manifest.json').read_text())['items']}
    results=json.loads((OUT/'qwen_results.json').read_text());assert len(results)>0 and len({(x['id'],x['condition']) for x in results})==len(results)
    protocol=json.loads((OUT/'qwen_protocol.json').read_text());assert len(protocol['conditions'])==16
    for item in inputs:
        rows=[x for x in results if x['id']==item['id']];assert {x['condition'] for x in rows}<=set(protocol['conditions'])
        ao=permutation(item['id'],'action');jo=permutation(item['id'],'justification')
        for row in rows:
            assert row['source']==item['source'] and row['model']==protocol['model']['id']
            ev=manifest[item['id']]['conditions'][row['condition']]
            assert row['images']==[str(Path(p).relative_to(OUT)) for p in ev['images']] and row['attached_text']==ev['attached_text']
            assert row['action_order']==ao and row['justification_order']==jo and row['gold_original_idx']==ann[item['id']]['correct']
            for field,order in [('action',ao),('justification',jo)]:
                prediction=row[field];choice=prediction['choice'];mapped=order[choice] if choice is not None else None
                parsed=re.fullmatch(r'\s*([1-5])\s*',prediction['text'])
                assert choice==(int(parsed.group(1))-1 if parsed else None)
                assert row[field+'_original_idx']==mapped and row[field+'_correct']==(mapped==ann[item['id']]['correct'])
                if prediction['prompt'] is not None:assert prediction['prompt'].startswith(protocol['prefix']+'\n'+row['attached_text'])
            assert row['both_correct']==(row['action_correct'] and row['justification_correct'])
            mode=row['condition'].rsplit('_',1)[1];assert ('Attached description:' in row['attached_text'])==(mode=='descriptions')
            sizes=[]
            for path in row['images']:
                with Image.open(OUT/path) as image:image.load();sizes.append(image.size)
            assert len(set(sizes))==1
    agent=json.loads((OUT/'gpt56-agent-results.json').read_text());assert len(agent['items'])==30
    pilot=json.loads((OUT/'gpt56-expanded-blind-manifest.json').read_text());pilots={x['id']:x for x in pilot['items']};views=0
    assert Counter(x['condition'] for x in agent['items'])==Counter({c:5 for c in ['rgb','rgb_descriptions','pose_descriptions','segmentation_descriptions','depth_descriptions','all_descriptions']})
    for row in agent['items']:
        assert 1<=row['action_position']<=5 and 1<=row['justification_position']<=5
        ev=row['image_view_evidence'];ev=[ev] if isinstance(ev,dict) else ev
        assert all(x['viewed'] and Path(x['path']).exists() for x in ev);views+=len(ev)
        if row['condition']!='rgb':
            expected=pilots[row['id']]['conditions'][row['condition']];assert [x['path'] for x in ev]==expected['images']
            ref=row['attached_text_source'];node=json.loads(Path(ref['manifest_path']).read_text())
            for key in ref['json_pointer'].strip('/').split('/'):node=node[int(key)] if isinstance(node,list) else node[key]
            assert node==expected['attached_text']
    # The user's subsequent failure-diagnostic request extends the original pilot.
    extended=json.loads((OUT/'gpt56-next10-rgb-results.json').read_text())['items']
    followup=json.loads((OUT/'gpt56-followup-results.json').read_text())
    followup_manifest=json.loads((OUT/'gpt56-followup-blind-manifest.json').read_text())['items']
    expected_followup={(x['id'],c) for x in followup_manifest for c in x['conditions']}
    assert len(extended)==10 and len(followup['items'])==60
    assert {(x['id'],x['condition']) for x in followup['items']}==expected_followup
    failure_summary=json.loads((OUT/'gpt56-failure-summary.json').read_text())
    baselines=[x for x in agent['items'] if x['condition']=='rgb']+extended
    def agent_correct(row):
        ao=permutation(row['id'],'action');jo=permutation(row['id'],'justification')
        return ao[row['action_position']-1]==jo[row['justification_position']-1]==ann[row['id']]['correct']
    assert failure_summary['rgb_baseline']['n']==15
    assert failure_summary['rgb_baseline']['both_correct']==sum(agent_correct(x) for x in baselines)
    assert {x['id'] for x in followup_manifest}=={x['id'] for x in extended if not agent_correct(x)}
    for c,data in failure_summary['conditions'].items():
        rr=[x for x in followup['items'] if x['condition']==c]
        assert data['n']==4 and data['both_correct']==sum(agent_correct(x) for x in rr)
    extra_links=Links();extra_links.feed((OUT/'gpt56-failures.html').read_text())
    for path in extra_links.paths:assert (OUT/path).exists(),path
    summary=json.loads((OUT/'summary.json').read_text())
    complete_ids=[x['id'] for x in inputs if {r['condition'] for r in results if r['id']==x['id']}==set(protocol['conditions'])]
    assert summary['complete_item_ids']==complete_ids and summary['saved_rows']==len(results)
    for c,metrics in summary['scores'].items():
        rr=[x for x in results if x['condition']==c]
        rr=[x for x in rr if x['id'] in summary['complete_item_ids']]
        for metric,data in metrics.items():assert data['n']==summary['items'] and data['correct']==sum(x[metric] for x in rr)
    parser=Links();parser.feed((OUT/'review.html').read_text())
    for path in parser.paths:assert (OUT/path).exists(),path
    script=re.search(r'<script>(.*?)</script>',(OUT/'review.html').read_text(),re.S)
    assert script
    subprocess.run(['node','--check'],input=script.group(1),text=True,check=True)
    segment=OUT/'idea5-expanded-section.tex';assert segment.exists() and 'author review required' in segment.read_text().lower()
    geo_protocol=json.loads((OUT/'geometry_protocol.json').read_text());assert hashlib.sha256((ROOT/'models/yolov8n-seg.pt').read_bytes()).hexdigest()==geo_protocol['segmentation_sha256']
    config=json.loads((ROOT/'models/Qwen3-VL-32B-Instruct-8bit/config.json').read_text());assert config['quantization']['bits']==8 and config['text_config']['num_hidden_layers']==64
    report={'status':'Stopped partial run; all saved rows checked','matched_complete_clips':summary['items'],'excluded_incomplete_rows':summary['excluded_incomplete_rows'],'qwen_results':len(results),'conditions':len(protocol['conditions']),'distinct_source_clips':40,'finite_depth_arrays':200,'geometry_images_decoded':decoded,'aligned_four_grid_sets':40,'agent_rows':30,'agent_image_views':views,'html_local_links_checked':len(parser.paths),'scores_recalculated':True,'mapped_choice_and_gold_consistency':True,'attached_description_manifest_consistency':True,'segmentation_weight_sha256_verified':True,'qwen_config_layers':64,'qwen_config_quantization_bits':8,'limits':'Asset, scoring and provenance checks do not validate detector accuracy, caption truth, student interpretation or TA approval.'}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
