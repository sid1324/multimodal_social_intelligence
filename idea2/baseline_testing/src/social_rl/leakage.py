"""Mandatory external-SFT checks against all official EgoNormia source IDs."""
import json
from .common import STORE,read_jsonl,sha256,write_json

def check_training(rows,run):
 root=STORE/'data/full_study';boundary=json.load(open(root/'boundary.json'))
 idsfile=root/'ego_source_ids.txt'
 if sha256(idsfile)!=boundary['ego_source_ids_sha256']:raise ValueError('Changed EgoNormia exclusion IDs')
 sources=set(idsfile.read_text().splitlines());assert len(sources)==boundary['ego_source_count']
 qids=[x['id'] for x in rows]
 if len(qids)!=len(set(qids)) or any(x['split']!='train' for x in rows):raise ValueError('Invalid training manifest')
 forbidden=[]
 if rows[0]['dataset']=='r3':
  forbidden=[x['id'] for x in rows if x['video_id'] in sources]
 elif rows[0]['dataset']=='psychemme':
  raw=STORE/'raw/psychemme/3b80a47473802b160b54e760d5dca070efd96f4c'
  qs={x['id']:x for x in read_jsonl(raw/'metadata/questions.jsonl')}
  ep={x['episode_id']:x for x in read_jsonl(raw/'metadata/episodes.jsonl')}
  eligible=set((STORE/'data/psychemme/audit/eligible_sft_ids.txt').read_text().splitlines())
  for x in rows:
   if x['id'] not in eligible or ep[qs[x['id']]['episode_id']]['source_video_id'] in sources:forbidden.append(x['id'])
 else:raise ValueError('Only R3 and PsycheMME are authorized training sources')
 report={'examples':len(rows),'excluded_ego_source_videos':len(sources),'detected_overlap_ids':forbidden,'ego_source_ids_sha256':sha256(idsfile),'scope':'Full official EgoNormia source-video exclusions; source metadata cannot certify unknown pretraining contamination.'}
 write_json(run/'leakage_audit.json',report)
 if forbidden:raise ValueError(f'EgoNormia overlap in training: {forbidden[:5]}')
