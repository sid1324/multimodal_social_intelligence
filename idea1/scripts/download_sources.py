"""Restore large public source assets/weights, pinned where the source permits it.
Default restores core tabular inputs. --media restores 400 source strips;
--models restores frozen MPNet/CLIP weights. No paid API calls or credentials.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,requests,hashlib,json,concurrent.futures,zipfile
p=argparse.ArgumentParser();p.add_argument('--external',action='store_true');p.add_argument('--media',action='store_true');p.add_argument('--models',action='store_true');p.add_argument('--normlens-images',action='store_true');a=p.parse_args()
def get(url,path,expected=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  r=requests.get(url,timeout=300);r.raise_for_status();path.write_bytes(r.content)
 if expected:assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, str(path)+' hash mismatch'
 return str(path)
# Source data is restored locally and is excluded from Git.
rev='09d8a7c53f06ed582236722ba8496c6a504cd1ab'
for name in ['final_data.json','final_data_eval.json','verified_split.json']:
 get(f'https://raw.githubusercontent.com/Open-Social-World/EgoNormia/{rev}/src/final_dataset/{name}',ROOT/'data/egonormia'/name,json.load(open(ROOT/'qa/provenance.json'))['source_hashes']['data/egonormia/'+name])
if a.media:
 manifests=json.load(open(ROOT/'outputs/logs/media_downloads.json'))
 with concurrent.futures.ThreadPoolExecutor(8) as e:
  for r in e.map(lambda m:get(m['url'],ROOT/'data/media'/(m['item']+'.jpg'),m['sha256']),manifests):print(r)
if a.normlens_images or a.external:
 get('https://media.githubusercontent.com/media/wade3han/normlens/2663c3240ac2f325950f5b4fe36ae196ed4d46be/normlens_dataset.zip',ROOT/'data/external/normlens_dataset.zip','b92f190bca09b748f97a94b2fe058908a93fa3e4ca0025d90f324cd0ed08eba7')
if a.external:
 provenance=json.load(open(ROOT/'qa/provenance.json'))
 get(provenance['normbank_url'],ROOT/'data/external/NormBank.csv',provenance['normbank_sha256'])
 get('https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/2937df7fa96d8515417e7b5122fbf8eaeeaeee86/train-norm-updated.parquet',ROOT/'data/external/egonormia_hf.parquet')
 with zipfile.ZipFile(ROOT/'data/external/normlens_dataset.zip') as archive:
  for name in ['high_agreement.jsonl','mid_agreement.jsonl']:
   matches=[n for n in archive.namelist() if Path(n).name==name and not n.startswith('__MACOSX/')]
   assert len(matches)==1, (name,matches)
   (ROOT/'data/external'/name).write_bytes(archive.read(matches[0]))
 hashes=json.load(open(ROOT/'qa/restored_source_hashes.json'))
 for rel,expected in hashes.items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected, rel+' hash mismatch'
if a.models:
 from huggingface_hub import snapshot_download
 for name in ['all-mpnet-base-v2','clip-vit-base-patch32']:
  m=json.load(open(ROOT/'outputs/logs'/('model_'+name+'.json')))
  snapshot_download(m['repo'],revision=m['revision'],local_dir=ROOT/'models'/name,allow_patterns=['*.json','*.safetensors','*.bin','*.txt','*.model'],ignore_patterns=['onnx/*','openvino/*'])
print('Source restoration complete.')
