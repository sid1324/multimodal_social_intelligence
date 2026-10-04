"""Read-only verification of all packaged final PNGs; no trait-validity claim."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]
counts={'frames':0,'editable':0,'copied':0,'outside_mask_changed_pixels':0}
for item in [969,1625,605,749,1250]:
 for method in ['omi']+(['sob_reference'] if item==969 else []):
  for condition in ['low','baseline','high']:
   for frame in range(1,6):
    a=np.asarray(Image.open(root/f'item_{item}/original/frame_{frame}.png').convert('RGB'))
    b=np.asarray(Image.open(root/f'item_{item}/{method}/{condition}/frame_{frame}.png').convert('RGB'))
    assert a.shape==b.shape, 'Unexpected dimensions'
    mp=root/f'item_{item}/masks/frame_{frame}.png'
    mask=np.asarray(Image.open(mp).convert('L'))>0 if mp.exists() else np.zeros(a.shape[:2],bool)
    change=np.any(a!=b,axis=2)
    outside=int(np.count_nonzero(change&~mask))
    assert outside==0, f'Changes outside mask: {item}/{method}/{condition}/{frame}'
    if mask.any():assert change.any(), 'Editable output is entirely unchanged'
    else:assert not change.any(), 'No-face frame changed'
    counts['frames']+=1;counts['editable']+=int(mask.any());counts['copied']+=int(not mask.any());counts['outside_mask_changed_pixels']+=outside
assert (counts['frames'],counts['editable'],counts['copied'])==(90,84,6)
print(json.dumps({'passed':True,**counts,'scope':'Pixel confinement only; not identity, realism or trait validity.'},indent=2))
