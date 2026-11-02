"""Keep original released frames and avoid reconstructing unsupported times."""
import tarfile
from pathlib import Path
from social_rl.common import STORE,read_jsonl,write_jsonl,write_json
root=STORE/'data/full_study';rows=read_jsonl(root/'psy_all.jsonl');needed=set()
for row in rows:
 episode=Path(row['frames'][0]).parent.name
 row['frames']=[str(STORE/'data/psychemme/media/frames'/episode/f'{i:03d}.jpg') for i in range(32)]
 row['visual_format']='ordered_images';needed.update(row['frames'])
archive=STORE/'raw/psychemme/3b80a47473802b160b54e760d5dca070efd96f4c/media/frames.tar'
with tarfile.open(archive) as tar:
 for m in tar:
  p=STORE/'data/psychemme'/m.name.removeprefix('./')
  if str(p) not in needed or p.exists():continue
  if not m.isfile() or '..' in Path(m.name).parts:raise ValueError(m.name)
  p.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(m) as src,p.open('wb') as dst:dst.write(src.read())
assert all(Path(p).exists() for p in needed)
write_jsonl(root/'psy_ordered_images32.jsonl',rows)
write_json(root/'frame_images_preparation.json',{'examples':len(rows),'frames_per_example':32,'unique_frames':len(needed),'timestamps':'None invented; released chronological frame images retained'})
print('Prepared',len(rows),'examples',len(needed),'frames',flush=True)
