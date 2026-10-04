from pathlib import Path
import json,shutil,hashlib,zipfile,html
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/workspace/scratch/8eb17a085ce3')
WORK=ROOT/'tmp/full_sequence_tests'
OUT=ROOT/'output/analysis/Full_Sequence_Face_Tests'
ITEMS=[969,1625,605,749,1250]
CONDITIONS=['low','baseline','high']
manifest=json.loads((WORK/'source_pilot_manifest.json').read_text())
notes={
969:'Glasses geometry and mouth posture vary across conditions. Original hair outline survives outside the mask; donor identity transfer is unverified. Hold for accessory/expression review.',
1625:'Registration fixed large misplaced patches, but forehead/hair seams remain in frames 2 and 5. Frame 3 offers only a tiny occluded cheek patch. Reject current sequence for a controlled experiment.',
605:'All five target regions are populated; downward poses remain broadly similar. Brows, eyelids and gaze differ across conditions, and donor transfer is unverified. Candidate for blinded review only.',
749:'All four visible faces are populated after registration; frame 1 is unchanged. High-condition mouths become more smile-like, particularly frames 3-5. Hold for expression control.',
1250:'A synthetic face replaces the target blur in every frame. Smiling increases in the high condition, and a patch boundary is visible near the lower face in frame 5. Hold for expression and edge repair.'}
sob_note='Published Figure 1 references were supplied, but outputs remain visually close to the original scene person. Reliable donor transfer and low-to-high ordering are unconfirmed. This does not test the authors\' generator.'

for item in ITEMS:
 for f in range(1,6):
  dest=OUT/f'item_{item}'/'original';dest.mkdir(parents=True,exist_ok=True)
  shutil.copy2(WORK/f'item_{item}_frame_{f}.png',dest/f'frame_{f}.png')
  native=OUT/f'item_{item}'/'native_source';native.mkdir(exist_ok=True)
  shutil.copy2(WORK/f'item_{item}_frame_{f}_source.png',native/f'frame_{f}.png')
 shutil.copy2(WORK/f'item_{item}_board.png',OUT/f'item_{item}'/'source_board.png')
(OUT/'data').mkdir(exist_ok=True)
shutil.copy2(WORK/'source_pilot_manifest.json',OUT/'data/source_pilot_manifest.json')
source_ann=ROOT/'output/analysis/MMML_Full_Screening/Chunk3_Pilots/EgoNormia_Chunk3_Pilots/data/source_annotations_selected.json'
shutil.copy2(source_ann,OUT/'data/source_annotations_selected.json')
(OUT/'references').mkdir(exist_ok=True)
for donor in [1340,1437,1284,1320]:
 for c in range(3):
  p=ROOT/f'tmp/face_method_research/assets/omi_{donor}_trustworthy_{c}.jpg';shutil.copy2(p,OUT/'references'/p.name)
for c in CONDITIONS:shutil.copy2(WORK/f'sob_published_{c}.png',OUT/'references'/f'sob_published_{c}.png')
for name in ['omi_descriptive_stats.json','omi_ranked_trust.json','omi_tree.json']:
 shutil.copy2(ROOT/'tmp/face_method_research'/name,OUT/'data'/name)

(OUT/'review').mkdir(exist_ok=True)
for p in (WORK/'qa').glob('*.png'):shutil.copy2(p,OUT/'review'/p.name)
audits=[]
for item in ITEMS:
 for method in ['omi']+(['sob_reference'] if item==969 else []):
  for c in CONDITIONS:
   for f in range(1,6):
    a=np.asarray(Image.open(OUT/f'item_{item}'/'original'/f'frame_{f}.png').convert('RGB')).astype(np.int16)
    p=OUT/f'item_{item}'/method/c/f'frame_{f}.png';b=np.asarray(Image.open(p).convert('RGB')).astype(np.int16)
    mp=OUT/f'item_{item}'/'masks'/f'frame_{f}.png'
    m=np.asarray(Image.open(mp).convert('L'))>0 if mp.exists() else np.zeros(a.shape[:2],bool)
    changed=np.any(a!=b,axis=2);outside=int(np.count_nonzero(changed&~m))
    if outside:raise ValueError(f'Outside-mask changes: {p}')
    diff=np.max(np.abs(b-a),axis=2);heat=np.zeros_like(a,dtype=np.uint8);heat[:,:,0]=np.clip(diff*4,0,255);heat[:,:,1]=np.clip(diff,0,255)
    dp=OUT/f'item_{item}'/'differences'/method/c;dp.mkdir(parents=True,exist_ok=True);Image.fromarray(heat).save(dp/f'frame_{f}.png')
    audits.append(dict(item=item,method=method,condition=c,frame=f,editable=bool(m.any()),changed_pixels=int(changed.sum()),outside_mask_changed_pixels=outside,mask_pixels=int(m.sum()),file=str(p.relative_to(OUT))))
  # Colored mask overlays are inspection diagrams, never study inputs.
 for f in range(1,6):
  im=Image.open(OUT/f'item_{item}'/'original'/f'frame_{f}.png').convert('RGB');mp=OUT/f'item_{item}'/'masks'/f'frame_{f}.png'
  if mp.exists():
   arr=np.asarray(im).astype(float);m=np.asarray(Image.open(mp).convert('L'))/255;arr=arr*(1-m[:,:,None]*.45)+np.array([0,180,255])*m[:,:,None]*.45;im=Image.fromarray(np.round(arr).astype('uint8'))
  dest=OUT/f'item_{item}'/'mask_overlays';dest.mkdir(exist_ok=True);im.save(dest/f'frame_{f}.png')
assert len(audits)==90 and sum(x['editable'] for x in audits)==84
summary=dict(variant_frames=90,edited_face_instances=84,unchanged_no_face_copies=6,original_sampled_frames=25,sequence_conditions=18,raw_generation_calls=19,outside_mask_changed_pixels=sum(x['outside_mask_changed_pixels'] for x in audits),working_resolution='640x360; item 605: 640x480',status='Feasibility outputs only; none validated for trait ordering or donor identity')
(OUT/'audit.json').write_text(json.dumps(dict(summary=summary,frames=audits),indent=2))

# Make retained execution scripts usable from an extracted package.
(OUT/'code').mkdir(exist_ok=True)
for name in ['register.py','qa_grid.py']:
 shutil.copy2(WORK/name,OUT/'code'/name)
js=(WORK/'node/composite.mjs').read_text().replace("const root='/workspace/scratch/8eb17a085ce3';", "const root=path.resolve(process.env.FACE_TEST_PACKAGE || '.');").replace("const work=path.join(root,'tmp/full_sequence_tests');", "const work=root;").replace("const out=path.join(root,'output/analysis/Full_Sequence_Face_Tests');", "const out=root;").replace("path.join(work,'source_pilot_manifest.json')", "path.join(work,'data/source_pilot_manifest.json')").replace("path.join(work,`item_${item}_frame_${f}.png`)", "path.join(work,`item_${item}`,'original',`frame_${f}.png`)")
js=js.replace('fs.copyFileSync(generatedPath,rawSaved);','if(path.resolve(generatedPath)!==path.resolve(rawSaved))fs.copyFileSync(generatedPath,rawSaved);')
(OUT/'code/composite.mjs').write_text(js)
(OUT/'code/package.json').write_text(json.dumps({'private':True,'type':'module','dependencies':{'sharp':'0.34.5'}},indent=2))
records=json.loads((OUT/'generation_records.json').read_text())
for r in records:
 r['reference_files']=[f'item_{r["item"]}/source_board.png',('references/'+Path(r['references'][1]).name)]
 r['raw_generated_file']=f'raw_generated/item_{r["item"]}_{r["method"]}_{r["condition"]}.png'
(OUT/'generation_records.json').write_text(json.dumps(records,indent=2))

readme='''# Full-sequence face-insertion feasibility tests

Start with **review.html** in this folder or **Full_Sequence_Test_Results.pdf**.
The HTML works offline after extracting the ZIP. It shows every original,
condition, enlarged face crop grid, mask and difference image.

## What was completed

- Five sampled pre-action frames in each of five selected clips: 25 originals.
- OMI: 5 clips x 5 frames x 3 intended trustworthiness conditions = 75 variants.
- Sobieszek published-reference test: item 969 x 5 frames x 3 conditions = 15 variants.
- Total: 90 variant frames, 84 edited face instances and 6 unchanged copies.
- 19 image-generation calls: 18 retained conditions and one superseded tabletop baseline.
- This scope does not include every frame of the original videos or all 1,853 examples.
- Only trustworthiness reference variants were tested. Dominance was not tested here.

## Exact process, in simple terms

1. Split each released five-frame preview into its five source frames.
2. Make working originals 640 pixels wide (360 high, except item 605: 480).
3. Put the five frames on one contact sheet and supply a synthetic donor reference.
4. Generate one five-frame draft per condition, requesting the same donor throughout.
5. Match background features outside the reviewed face rectangle to estimate a projective
   alignment for every visible face frame. Resample the draft in JavaScript into the
   source frame's coordinate system. Background matching does not prove facial alignment.
6. Blend the aligned draft only inside the same fixed mask in every condition.
   Copy source pixels exactly wherever mask intensity is zero. Mask blur has a small halo.
7. Decode all final PNGs and independently check changes against the working originals.
8. Inspect all final face-region comparisons and record failures below.

## What passed and what did not

All 90 outputs exist. All pixels outside mask support are exactly unchanged relative
to the canonical working originals. The 6 no-face copies are entirely unchanged.
This does not establish that every pixel inside the mask belongs to facial skin:
glasses, hair, occluders, blur and seams can still appear inside the reviewed region.

No condition set is approved as an experiment-ready trait manipulation. Visible
expression, gaze, accessory, identity and seam changes remain. The published donor
ratings describe the donor references, not the generated scene insertions. A model
decision difference on these images cannot yet be attributed solely to physiognomy.

The original unregistered mask integration failed in several scenes; diagnostic
grids are retained. All current conditions use background registration. The initial
OMI tabletop baseline used a different prompt; it was superseded by a baseline
using the same prompt family as low/high, and retained under diagnostics.

## Review findings by clip

'''
for item in ITEMS:readme+=f'- Item {item}: {notes[item]}\n'
readme+=f'- Additional Sobieszek reference test: {sob_note}\n'
readme+='''
## Scientific limits and recommended next chunk

Run a blinded stimulus-quality review before VLM action tests. Rate perceived
trustworthiness, same identity, realism, expression/gaze stability and visible
artifacts. Decide in advance what failure excludes a sequence. Repair accessory
and expression failures or use a more constrained, reproducible face-transfer
pipeline. This review has not been conducted and no human ratings are claimed.

No VLM action inference, bias measurement, DPO, mitigation training or human
recruitment was performed. The SFT pipeline was not run. The Sobieszek comparison
uses cropped published Figure 1 references and the same generative compositor;
it is not execution or validation of their generator. Its archive/ratings identity
mapping was not assumed. Neither method here is a direct paste of unchanged dataset pixels.

## Donors and provenance

OMI assignments: 969 -> 1340; 1625 -> 1437; 605 -> 1284; 749 -> 1284; 1250 -> 1320.
Donor 1284 was reused across two clips. Thus five scenes do not provide five
independent donor identities. All OMI donors used are synthetic.
Released levels 0/1/2 are used as intended low/baseline/high references. Selected
donors were screened by first-presentation published rating means; those means
are not ratings of these composites. Selection preceded action-model outcomes.

OMI source: https://github.com/jcpeterson/omi
Pinned reference repository commit: 53bb3c13113afc3ae67aa6ede4ac9dfe33f306af.
OMI code/data license: CC BY-NC-SA 4.0. Retain the authors' terms and attribution.

Sobieszek et al., Enhancing control over facial stimuli: A revised method of
manipulating perceived trustworthiness and dominance of faces (2026 issue;
online 2024), DOI: https://doi.org/10.1111/bjop.12732.
Reference faces: lower-row revised-method triplet in published Figure 1;
source image BJOP-117-636-g004.jpg. Published article license: CC BY-NC 4.0.
Author materials: https://github.com/AdamSobieszek/psychGAN and https://osf.io/du5m8/.
These are reference-based derivatives for research review. No broader license
claim is made for the original video-preview material or full package.

## Rechecking the invariant

With Python, Pillow and NumPy installed, run `python code/verify.py` from the
extracted folder. It reads the current PNGs and checks all 90 frames, including
unchanged copies. `audit.json` includes the per-frame numbers.

`code/composite.mjs` reproduces the final deterministic integration from retained
raw boards, masks/region definitions and registration.json. Install Sharp from
code/package.json, set FACE_TEST_PACKAGE to the extracted folder and run:
`node code/composite.mjs 969 omi low raw_generated/item_969_omi_low.png`.
It overwrites the selected condition locally. Work on a copy if preserving the
archive is desired. Retained register.py and qa_grid.py record the original
execution analysis; they contain original workspace paths and need adaptation.
Image generation is stochastic: prompts and references are retained, but exact
regeneration of donor morphology is not guaranteed. No model version or random
seed was exposed by the image tool, so neither is invented.

Difference images show max absolute RGB difference x4 in red (green unamplified),
not model saliency. Colored mask overlays are inspection aids, never study inputs.
Native-source frames are included; pixel-invariance claims use working originals.
'''
(OUT/'README.md').write_text(readme)
(OUT/'GENERATION_STATUS.md').write_text('# Completion status\n\nAll 18 retained condition sequences complete; 90 final registered masked frames.\n\nNone validated for causal trait comparisons. See README.md and audit.json.\n')

pages=[]
for item in ITEMS:
 for method in ['omi']+(['sob_reference'] if item==969 else []):
  pages.append(dict(item=item,method=method,note=notes[item] if method=='omi' else sob_note,grid=f'review/item_{item}_{method}_crops.png'))
viewer='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-sequence face tests</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#172435}main{max-width:1350px;margin:auto}h1{margin-bottom:8px}select,button{font:inherit;padding:8px;margin:6px}img{max-width:100%;background:white}section{background:white;border-radius:10px;padding:18px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.warn{border-left:5px solid #d68b16;padding:12px;background:#fff3db}p{line-height:1.5}.grid{display:block;max-height:900px;margin:auto}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>
<main><h1>All five sampled frames: face-insertion tests</h1><p>25 working originals · 90 variant frames · 84 edited face instances · 6 unchanged copies.</p>
<p class="warn">Feasibility outputs. Outside-mask preservation passes; identity, perceived trait ordering and expression control are unvalidated. Read <a href="README.md">the complete process and findings</a>.</p>
<section><label>Sequence <select id="sequence"></select></label><label>Frame <select id="frame"><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label><label>View <select id="condition"><option value="low">Low reference</option><option value="baseline">Baseline reference</option><option value="high">High reference</option><option value="mask">Mask overlay</option><option value="difference">High minus original difference</option></select></label><p id="note"></p><div class="pair"><div><h3>Working original</h3><a id="originalLink"><img id="original"></a></div><div><h3 id="outputLabel">Selected output</h3><a id="outputLink"><img id="output"></a></div></div></section><section><h2>All five face regions</h2><p>Rows: original, low, baseline, high. Columns: frames 1-5. Enlargement adds no recovered detail.</p><img id="grid" class="grid"></section><p>Images can be opened at full size by clicking. Every condition's difference image and pixel check is in its folder. This offline viewer sends no data.</p></main><script>
const scenes=SCENES;const seq=document.getElementById('sequence');scenes.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`Item ${s.item} / ${s.method}`;seq.append(o)});
function update(){let s=scenes[Number(seq.value)],f=document.getElementById('frame').value,c=document.getElementById('condition').value;let original=`item_${s.item}/original/frame_${f}.png`;let output=c==='mask'?`item_${s.item}/mask_overlays/frame_${f}.png`:c==='difference'?`item_${s.item}/differences/${s.method}/high/frame_${f}.png`:`item_${s.item}/${s.method}/${c}/frame_${f}.png`;document.getElementById('original').src=original;document.getElementById('originalLink').href=original;document.getElementById('output').src=output;document.getElementById('outputLink').href=output;document.getElementById('note').textContent=s.note;document.getElementById('grid').src=s.grid;document.getElementById('outputLabel').textContent=document.getElementById('condition').selectedOptions[0].textContent;}
['sequence','frame','condition'].forEach(id=>document.getElementById(id).addEventListener('change',update));update();</script></html>'''.replace('const scenes=SCENES','const scenes='+json.dumps(pages))
(OUT/'review.html').write_text(viewer)
(OUT/'review_findings.json').write_text(json.dumps(pages,indent=2))
print(json.dumps(summary))
