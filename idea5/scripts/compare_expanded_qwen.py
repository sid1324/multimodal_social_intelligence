"""Expanded frozen-model, paired image/description ablations on 40 EgoNormia clips."""
import hashlib
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

from compare_qwen_pose import permutation

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/idea5-audit';OUT=ROOT/'output/idea5-expanded'
PREFIX='''You observe a situation from an egocentric camera. Every image is a chronological grid of five context frames, read left to right, sampled before a next action. RGB shows the observed scene. Pose maps show estimated body landmarks in yellow and hand landmarks in magenta. Segmentation maps show predicted COCO-class instance masks and labels; black is unsegmented, not empty space. Depth maps show estimated relative inverse depth: brighter suggests nearer, darker farther; they do not give metric distances. Maps and any attached generated descriptions can be incomplete, inaccurate, or inconsistent. Judge the observed context rather than assuming an auxiliary representation is correct.
'''
BASE_CONDITIONS={'rgb':['rgb'],'pose':['rgb','pose'],'segmentation':['rgb','segmentation'],'depth':['rgb','depth'],'all':['rgb','pose','segmentation','depth'],'repeat1':['rgb','rgb'],'repeat3':['rgb','rgb','rgb','rgb'],'shuffled_all':['rgb','pose','segmentation','depth']}

def evidence(item,base,text,donor=None):
    image_paths=[];sections=[];role_names=BASE_CONDITIONS[base]
    for i,role in enumerate(role_names):
        source=donor if base=='shuffled_all' and role!='rgb' else item
        image=OUT/source['images'][role]
        if source is donor:
            destination=OUT/'grids'/f"{item['id']}_shuffled_{role}.jpg"
            with Image.open(OUT/item['images']['rgb']) as target,Image.open(image) as original:
                original.resize(target.size,Image.Resampling.NEAREST).save(destination,quality=95)
            image=destination
        image_paths.append(str(image))
        section=f'Image {i+1}: {role}.'
        if text:section+='\nAttached description: '+source['descriptions'][role]
        sections.append(section)
    return image_paths,'\n\n'.join(sections)

def main():
    inputs=json.loads((OUT/'inputs.json').read_text());items=inputs['items'];assert len(items)==40 and all(x['descriptions'].get('rgb') for x in items)
    model_info=json.loads((OUT/'model.json').read_text());model_path=str(ROOT/model_info['local_path'])
    sample={i['id']:i for i in json.loads((OLD/'protocol.json').read_text())['sample']}
    conditions=[f'{base}_{mode}' for mode in ['images','descriptions'] for base in BASE_CONDITIONS]
    protocol={'model':model_info,'sample_ids':[x['id'] for x in items],'conditions':conditions,'prefix':PREFIX,'temperature':0.0,'seed':42,'max_tokens':16,'responses':'Action selection followed by justification selection conditioned on predicted action. Strict 1-5 integer output; invalid counted wrong.','option_order':'Same independently permuted action/justification options as original experiment, held fixed across all new conditions.','descriptions':{'rgb':inputs['rgb_caption_generation'],'processed':inputs['processed_description_generation']},'controls':'repeat1 matches one auxiliary image slot; repeat3 matches three slots. Same RGB grid resolution in every condition. Shuffled-all uses next source cyclically and resizes maps to target grid dimensions; associated processed descriptions are also shuffled. Text length is not matched across conditions; image-only versus description conditions separates added text.','leakage':'No dataset descriptions, gold, or taxonomy enter model prompts. RGB caption generation sees RGB only and no answer options.','inference_setup':'Local stateless generate call per prompt, no chat history, no fine-tuning; separate 32B run rather than pooled with earlier 8B pilot.'}
    (OUT/'qwen_protocol.json').write_text(json.dumps(protocol,indent=2))
    # Gold-free manifests allow another agent to see identical inputs and choices.
    manifests=[]
    for j,item in enumerate(items):
        ann=sample[item['id']]['annotation'];ao=permutation(item['id'],'action');jo=permutation(item['id'],'justification')
        row={'id':item['id'],'source':item['source'],'action_options':[ann['behaviors'][i] or '[None of these actions is appropriate]' for i in ao],'justification_options':[ann['justifications'][i] or '[None of these justifications is appropriate]' for i in jo],'conditions':{}}
        for condition in conditions:
            base,mode=condition.rsplit('_',1);images,text=evidence(item,base,mode=='descriptions',items[(j+1)%len(items)])
            row['conditions'][condition]={'images':images,'attached_text':text,'prefix':PREFIX}
        manifests.append(row)
    (OUT/'qwen-first5-blind-manifest.json').write_text(json.dumps({'gold_free':True,'items':manifests[:5],'indexing':'1-based option positions','note':'Do not access root annotation/protocol/result files. Each condition uses the same options and source frames.'},indent=2))
    (OUT/'full-blind-manifest.json').write_text(json.dumps({'gold_free':True,'items':manifests},indent=2))
    model,processor=load(model_path);config=load_config(model_path)
    path=OUT/'qwen_results.json';results=json.loads(path.read_text()) if path.exists() else [];done={(x['id'],x['condition']) for x in results}
    def infer(prompt,images):
        mx.random.seed(42);formatted=apply_chat_template(processor,config,prompt,num_images=len(images));t=time.time()
        output=generate(model,processor,formatted,image=images,max_tokens=16,temperature=0.,verbose=False)
        response=output.text if hasattr(output,'text') else str(output);match=re.fullmatch(r'\s*([1-5])\s*',response)
        return {'text':response,'choice':int(match.group(1))-1 if match else None,'seconds':time.time()-t,'prompt':prompt,'stats':{k:getattr(output,k,None) for k in ['prompt_tokens','generation_tokens','prompt_tps','generation_tps','peak_memory']}}
    started=time.time()
    for j,item in enumerate(items):
        item_id=item['id'];ann=sample[item_id]['annotation'];ao=permutation(item_id,'action');jo=permutation(item_id,'justification');manifest=manifests[j]
        action_options='\n'.join(f"{k+1}. {text}" for k,text in enumerate(manifest['action_options']));just_options='\n'.join(f"{k+1}. {text}" for k,text in enumerate(manifest['justification_options']))
        for condition in conditions:
            if (item_id,condition) in done:continue
            ev=manifest['conditions'][condition];images=ev['images'];context=PREFIX+'\n'+ev['attached_text']
            ap=context+'\n\nWhich single action is most normatively appropriate for the camera wearer to perform next? Use the observed context, not just answer wording. The none option means no offered action is appropriate, not uncertainty. Return only one integer from 1 to 5, without explanation.\nActions:\n'+action_options
            action=infer(ap,images);aidx=ao[action['choice']] if action['choice'] is not None else None
            if aidx is not None:
                selected=ann['behaviors'][aidx] or 'None of the offered actions'
                jp=context+f'\n\nThe selected action is: {selected}\nWhich justification best supports that action in this context? The selected action may itself be inappropriate. Choose none if no justification supports it. Return only one integer from 1 to 5, without explanation.\nJustifications:\n'+just_options
                justification=infer(jp,images);jidx=jo[justification['choice']] if justification['choice'] is not None else None
            else:justification={'text':'','choice':None,'seconds':0,'stats':{},'prompt':None};jidx=None
            gold=ann['correct'];row={'id':item_id,'source':item['source'],'condition':condition,'model':model_info['id'],'images':[str(Path(p).relative_to(OUT)) for p in images],'attached_text':ev['attached_text'],'action_order':ao,'justification_order':jo,'action':action,'justification':justification,'action_original_idx':aidx,'justification_original_idx':jidx,'gold_original_idx':gold,'action_correct':aidx==gold,'justification_correct':jidx==gold,'both_correct':aidx==gold and jidx==gold}
            results.append(row);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(results,indent=2));temp.replace(path);mx.clear_cache()
            print(f"{len(results)}/{len(items)*len(conditions)} {condition}; action={aidx}, justification={jidx}, both={row['both_correct']}",flush=True)
    (OUT/'qwen_runtime.json').write_text(json.dumps({'seconds_this_invocation':time.time()-started,'conditions':len(conditions),'items':len(items),'results':len(results),'peak_memory_GB':max(x['action']['stats'].get('peak_memory') or 0 for x in results)},indent=2))
    print('Expanded 32B comparison complete',flush=True)

if __name__=='__main__':main()
