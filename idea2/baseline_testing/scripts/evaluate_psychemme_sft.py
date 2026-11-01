"""Evaluate a frozen adapter on source-video-grouped PsycheMME validation."""
import argparse
import json
import re
from pathlib import Path

from social_rl.common import STORE, append_jsonl, read_jsonl, write_json
from social_rl.model import VideoPolicy
from train_psychemme_sft import prompt_and_target


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=120)
    args=p.parse_args()
    rows=read_jsonl(STORE/'data/psychemme/val.jsonl')
    rows=sorted(rows,key=lambda x:x['id'])[:args.limit]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    policy=VideoPolicy(adapter=args.adapter)
    results=[]
    for i,row in enumerate(rows):
        prompt,target=prompt_and_target(row,777+i)
        expected=re.search(r'Answer: ([A-Z])$',target).group(1)
        inputs=policy.inputs(prompt,frame_paths=row['frames'])
        tokens,reply=policy.generate(inputs,max_new_tokens=128)
        found=re.findall(r'Answer:\s*([A-Z])\b',reply)
        answer=found[-1] if found else None
        result={'id':row['id'],'group':row['group'],'category':row['category'],
                'expected':expected,'predicted':answer,'correct':answer==expected,
                'response':reply,'output_tokens':tokens.numel()}
        append_jsonl(out/'predictions.jsonl',result);results.append(result)
    write_json(out/'summary.json',{'attempted':len(results),'correct':sum(x['correct'] for x in results),
                                    'accuracy':sum(x['correct'] for x in results)/len(results),
                                    'source_groups':len({x['group'] for x in results})})
    print((out/'summary.json').read_text())


if __name__=='__main__':main()
