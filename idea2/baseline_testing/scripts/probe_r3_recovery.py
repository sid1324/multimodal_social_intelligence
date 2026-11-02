"""Bounded serial retry of public R3 source videos after a long cooldown."""
import json,time
from social_rl.common import STORE,read_jsonl,write_json,append_jsonl
from acquire_media import fetch_r3
root=STORE/'data/full_study';coverage=json.load(open(root/'coverage.json'));ids=sorted({x['id'] for x in coverage['r3']['failures'] if 'Sign in' in x.get('error','')})[:5]
rows={x['video_id']:x for x in read_jsonl(root/'r3_all.jsonl')};results=[]
for uid in ids:
 result=fetch_r3(rows[uid]);results.append(result);append_jsonl(root/'r3_cooldown_probe.jsonl',result);print(json.dumps(result),flush=True);time.sleep(10)
write_json(root/'r3_cooldown_probe_status.json',{'attempted':len(results),'recovered':sum(x['status']=='ok' for x in results),'note':'Only public source retries; no login/cookie/CAPTCHA bypass. Existing training manifests unchanged.'})
