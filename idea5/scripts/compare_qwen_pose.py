"""Frozen Qwen RGB/pose pilot with matched extra-image controls."""
import hashlib
import json
import re
import time
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/idea5-audit"
MODEL_ID="mlx-community/Qwen3-VL-8B-Instruct-4bit"
PREFIX="""You observe a situation from an egocentric camera. Image 1 is a chronological grid of five RGB context frames, read left to right. If image 2 is present, it is an auxiliary grid: it can repeat the RGB frames or display estimated body (yellow) and hand (magenta) landmarks. Auxiliary landmarks may be missing, inaccurate, or unrelated. Use the RGB context to judge the situation; do not assume the auxiliary image is correct. No action has yet been taken.
"""

def permutation(item_id,kind):
    return sorted(range(5),key=lambda i:hashlib.sha256(f"qwen-pose-v1:{item_id}:{kind}:{i}".encode()).hexdigest())

def grid(records,kind,target):
    tiles=[]
    for r in records:
        im=Image.open(OUT/r["paths"][kind]).convert("RGB")
        im.thumbnail((384,384),Image.Resampling.LANCZOS)
        tiles.append(im)
    height=max(im.height for im in tiles); width=sum(im.width for im in tiles)
    canvas=Image.new("RGB",(width,height)); x=0
    for im in tiles:
        canvas.paste(im,(x,0)); x+=im.width
    canvas.save(target,quality=95)
    return str(target)

def main():
    import mlx.core as mx
    import mlx_vlm
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from huggingface_hub import HfApi,snapshot_download
    protocol=json.loads((OUT/"protocol.json").read_text())
    records=json.loads((OUT/"frame_results.json").read_text())
    grouped={i["id"]:sorted([r for r in records if r["id"]==i["id"]],key=lambda r:r["sample_index"]) for i in protocol["sample"]}
    folder=OUT/"grids";folder.mkdir(exist_ok=True)
    image_paths={}
    for item_id,frames in grouped.items():
        image_paths[item_id]={kind:grid(frames,kind,folder/f"{item_id}_{kind}.jpg") for kind in ["rgb","skeleton"]}
    previous_protocol_path=OUT/"qwen_protocol.json"
    revision=json.loads(previous_protocol_path.read_text())["revision"] if previous_protocol_path.exists() else HfApi().model_info(MODEL_ID).sha
    print(f"Downloading pinned model {MODEL_ID}@{revision}",flush=True)
    model_path=snapshot_download(MODEL_ID,revision=revision,local_dir=str(ROOT/"models/Qwen3-VL-8B-Instruct-4bit"))
    conditions=["rgb","rgb_repeat","rgb_pose","rgb_shuffled_pose"]
    run_protocol={"model":MODEL_ID,"revision":revision,"quantization":"community 4-bit checkpoint; not full-precision paper model","conditions":conditions,"temperature":0.0,"max_tokens":16,"seed":42,"action_then_justification":True,"independent_option_permutations":"SHA256 fixed per item and task, identical across conditions","frame_grid":"Five frames in a horizontal strip; each frame maximum side 384px; RGB resolution identical in every condition; auxiliary grids have same dimensions as RGB grids for that item.","shuffled_pose":"Next source in the preselected sample, cyclic rotation; resized to target grid dimensions.","prefix":PREFIX,"descriptions_or_gold_or_norm_labels_in_prompt":False,"evaluation":"Custom subset pilot, not official leaderboard replication; same sources as extraction audit."}
    (OUT/"qwen_protocol.json").write_text(json.dumps(run_protocol,indent=2))
    model,processor=load(model_path);config=load_config(model_path)
    results_path=OUT/"qwen_results.json"
    results=json.loads(results_path.read_text()) if results_path.exists() else []
    done={(r["id"],r["condition"]) for r in results}
    def infer(prompt,images):
        mx.random.seed(42)
        formatted=apply_chat_template(processor,config,prompt,num_images=len(images))
        started=time.time()
        output=generate(model,processor,formatted,image=images,max_tokens=16,temperature=0.0,verbose=False)
        text=output if isinstance(output,str) else output.text
        valid=re.fullmatch(r"\s*([1-5])\s*",text)
        choice=int(valid.group(1))-1 if valid else None
        stats={k:getattr(output,k,None) for k in ["prompt_tokens","generation_tokens","prompt_tps","generation_tps","peak_memory"]}
        return {"text":text,"choice":choice,"seconds":time.time()-started,"stats":stats,"prompt":prompt}
    for position,item in enumerate(protocol["sample"]):
        item_id=item["id"];annotation=item["annotation"];paths=image_paths[item_id]
        action_order=permutation(item_id,"action");justify_order=permutation(item_id,"justification")
        action_options="\n".join(f"{k+1}. {annotation['behaviors'][i] or '[None of these actions is appropriate]'}" for k,i in enumerate(action_order))
        justify_options="\n".join(f"{k+1}. {annotation['justifications'][i] or '[None of these justifications is appropriate]'}" for k,i in enumerate(justify_order))
        other_id=protocol["sample"][(position+1)%len(protocol["sample"])]["id"]
        mismatched_path=folder/f"{item_id}_shuffled_skeleton.jpg"
        with Image.open(paths["rgb"]) as rgb,Image.open(image_paths[other_id]["skeleton"]) as other:
            other.resize(rgb.size,Image.Resampling.NEAREST).save(mismatched_path,quality=95)
        for condition in conditions:
            if (item_id,condition) in done:continue
            images=[paths["rgb"]]
            if condition=="rgb_repeat":images.append(paths["rgb"])
            elif condition=="rgb_pose":images.append(paths["skeleton"])
            elif condition=="rgb_shuffled_pose":images.append(str(mismatched_path))
            action_prompt=PREFIX+"\nWhich single action is most normatively appropriate for the camera wearer to perform next? Judge normative appropriateness using the observed context. The none option means no offered action is appropriate, not uncertainty. Return only one integer from 1 to 5, with no explanation.\nActions:\n"+action_options
            action=infer(action_prompt,images)
            aidx=action_order[action["choice"]] if action["choice"] is not None else None
            if aidx is not None:
                selected=annotation["behaviors"][aidx] or "None of the offered actions"
                just_prompt=PREFIX+f"\nThe selected action is: {selected}\nWhich justification best supports that action in this context? The action may itself be inappropriate. Choose none if no offered justification supports it. Return only one integer from 1 to 5, with no explanation.\nJustifications:\n"+justify_options
                justification=infer(just_prompt,images)
                jidx=justify_order[justification["choice"]] if justification["choice"] is not None else None
            else:
                justification={"text":"","choice":None,"seconds":0,"stats":{},"prompt":None};jidx=None
            gold=annotation["correct"]
            result={"id":item_id,"source":item["source"],"categories":item["categories"],"condition":condition,"images":[str(Path(p).relative_to(OUT)) for p in images],"action_order":action_order,"justification_order":justify_order,"action":action,"justification":justification,"action_original_idx":aidx,"justification_original_idx":jidx,"gold_original_idx":gold,"action_correct":aidx==gold,"justification_correct":jidx==gold,"both_correct":aidx==gold and jidx==gold}
            results.append(result)
            temp=results_path.with_suffix(".tmp");temp.write_text(json.dumps(results,indent=2));temp.replace(results_path)
            mx.clear_cache()
            print(f"{len(results)}/{len(protocol['sample'])*len(conditions)} {condition}: action={aidx} justification={jidx} gold={gold}",flush=True)
    print("Completed frozen-model comparison",flush=True)

if __name__=="__main__":main()
