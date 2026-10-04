from pathlib import Path
import json,shutil,zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

ROOT=Path('/workspace/scratch/8eb17a085ce3');OUT=ROOT/'output/analysis/Milestone2_Consolidation'
d=json.loads((OUT/'consolidated_results.json').read_text());s=d['summary']
caption1='Figure 1. Original-data feasibility. (A) AI inspection of all five sampled frames in 117 purposively selected examples identified 49 promising regions, 51 requiring editing tests and 17 with no suitable region identified. The other 1,736 examples remain visually unverified. (B) Correct-option norm proportions use separate denominators for the 100 potential candidates and all 1,853 items; multi-label counts overlap. Differences describe the selected pool, not population prevalence or model bias. (C) Each point is one AI planning rectangle from the 23 visible target frames in five pilots; black ticks mark within-pilot medians. These are region footprints, not segmented skin areas or validated trait legibility. Frames within a clip are dependent.'
caption2='Figure 2. Technical construction and remaining failures. Original and final registered, fixed-mask OMI-reference composites for item 605/frame 3, item 1625/frame 2 and item 749/frame 3. The examples illustrate a small target region, a forehead/hair seam and expression drift. They were selected to display constraints, not estimate their frequency. Low/baseline/high label supplied references; perceived trait ordering and identity retention of the outputs have not been validated. Crops are enlarged from the canonical working originals and add no recovered detail.'
md='''# Final consolidation for Milestone 2

Analysis type: **Idea-specific / Data analysis**. This packet consolidates evidence
for the feasibility of Idea 3; it is not the final ICML team report.

## Premises examined

1. EgoNormia contains localizable partner-face regions that could support controlled insertion.
2. Reference-guided insertions can preserve the nonfacial scene across the five sampled frames.

These are the feasibility premises motivating the work, not a preregistered
statistical test. Our work tests construction and coverage, not Qwen's bias.
Do not present exploratory findings as confirmatory causal results.

## Verified results and scope

| Result | Value | Interpretation |
|---|---:|---|
| Original items screened automatically | 1,853 | All pinned official annotation IDs |
| Sampled frames screened automatically | 9,265 | Five frames per item; full videos not inspected |
| Distinct source-video ID prefixes | 1,075 | Computed from the pinned annotation snapshot |
| Screening/download failures | 0 | Does not establish detector accuracy |
| Five-frame AI visual review | 117 examples / 585 frames | Purposive selection, not a random sample |
| Promising insertion regions | 49 | Potential candidates, not validated stimuli |
| Need editing tests | 51 | Potential candidates retained despite constraints |
| No suitable region identified | 17 | Within the five reviewed frames only |
| Still visually unverified | 1,736 | Must not be labeled unusable |
| Potential pool | 100 examples / 92 source IDs | 49 + 51, with repeated scenes possible |
| Primary-norm union | 81 / 100 | Cooperation, communication, coordination or privacy |
| Editing pilots | 5 clips / 25 original frames | Five source IDs; all include Cooperation |
| Visible target-region plans | 23 / 25 frames | Includes one severely occluded cheek region |
| Retained variant frames | 90 | 75 OMI + 15 Sobieszek-reference composites |
| Edited face instances / unchanged copies | 84 / 6 | Tubing frame 1 and checkout frame 4 copied in 3 conditions |
| Outside-mask changed pixels | 0 | All 90 final frames; canonical working baseline |
| Validated trait-condition sets | None yet | Not a zero success-rate estimate |
| Qwen trials | 0 | No behavioral or trait-legibility result |

The v1 automatic triage was 69 strong suggestions, 1,430 uncertain and 354
with no region detected. Its one-frame AI audit (32 promising / 36 needing
review / 1 false detection) is historical. Use the v2 all-five-frame counts
above for current conclusions. V1 no-detection labels never prove face absence.

## Correct-option norm coverage

| Norm | Promising /49 | Needs test /51 | Pool /100 | Full /1853 | Pool % | Full % |
|---|---:|---:|---:|---:|---:|---:|
'''
for r in d['norm_coverage']:md+=f"| {r['norm']} | {r['promising']} | {r['needs_test']} | {r['pool']} | {r['full']} | {r['pool_percent']:.1f} | {r['full_percent']:.1f} |\n"
md+='''
Counts overlap because labels are multi-label. They refer to the correct option,
not every label mentioned by any option. Three pool items have no correct-option
labels: 721, 723 and 981. Item 721's gold action is the empty NONE-appropriate
sentinel; the other two have nonempty gold actions with empty taxonomies.

## Source and setting concentration

- Seven source IDs contribute 15 pool items; selecting one per source leaves 92 representatives.
- 25/100 candidates have a coarse shop-style counter/checkout setting label.
- The apparent stocked-fridge/counter cluster contains 24 candidates from 22 source IDs.
- The pool spans 11 coarse setting labels. Different IDs/labels do not prove independent people or sites.
- No identical full-preview byte groups were found in the prior check; near duplicates/events remain unverified.

These setting labels are AI annotations, not verified location or identity matches.
The source count computed here is 1,075, whereas the proposal quoted 1,077;
report the snapshot-specific computed count without inventing an explanation.

## Pilot summary

| Item | Situation | Target frames | Median review area % | Median mask support % | Shortest-side range (working px) | OMI outputs | Additional Sob outputs |
|---|---|---:|---:|---:|---:|---:|---:|
'''
for r in d['pilot_table']:md+=f"| {r['item']} | {r['setting']} | {r['visible_frames']}/5 | {r['median_region_area_percent']:.2f} | {r['median_mask_support_area_percent']:.2f} | {r['min_region_side_px']}-{r['max_region_side_px']} | {r['omi_outputs']} | {r['additional_sob_reference_outputs']} |\n"
md+='''
Across the 23 visible planning rectangles, median area is **3.19%** of the
working frame (range **0.93-21.25%**); shortest sides range **46-211 pixels**.
These statistics are for the five selected pilots only and are not a face-size
distribution for all 1,853 items. A review rectangle can contain occluders,
accessories, hair and surrounding blur; the smaller mask-support area includes
the feathering halo and is not a segmented face either.

Working originals: 640 x 360, except item 605 at 640 x 480. Native previews
were downsampled first. Pixel preservation is measured against these working
originals, not the native JPEG. Source selection preceded action-model outcomes.

## Pilot limitations

'''
for r in d['pilot_table']:md+=f"- **{r['item']}:** {r['limitations']}\n"
md+='''
- **Additional Sobieszek-reference test:** outputs remain close to the original scene person; donor transfer and trait ordering are unconfirmed. The authors' generator was not run.
- OMI donor 1284 was reused in items 605 and 749, so the five OMI scenes use four donor identities.
- Four selected original previews show facial detail; item 1250 has a blurred target. Revise the blanket claim that every released face is blurred.
- A synthetic face inserted into a blurred region does not reconstruct the original person's identity.

## Short findings for the write-up

1. A pool of 100 potential face-region examples was identified among 117 purposively reviewed examples.
2. Usable original facial detail is unnecessary for considering a region for synthetic insertion.
3. Candidate coverage is uneven: Privacy has five candidates, Coordination ten, and checkout-like scenes recur.
4. Region footprint varies considerably across the five pilots; it does not establish trait visibility to Qwen.
5. Fixed-mask integration preserves the surrounding working-original pixels exactly, but expressions, accessories and seams can still vary inside the mask.
6. The feasibility investigation supports continued development; it does not yet establish a controlled causal manipulation or model bias.

## Implications for Idea 3

- Retain potential blurred, angled and constrained regions; require editing tests before declaring eligibility.
- Sample deliberately across norms and reviewed scene/source groups; do not treat the five pilots as balanced coverage.
- Replace the blanket blur assumption with source-specific observations.
- Improve donor transfer, accessory preservation and expression/gaze control before the main experiment.
- Validate realism, same identity and intended perceived trait ordering in the final composites.
- Then run Qwen trait-legibility checks, followed by matched action-choice comparisons; only these later tests address model sensitivity/bias.
- Preserve the proposed text-cue comparison for later work; no text-arm trials or DPO training were executed here.

## Analysis requirements and remaining writing

| Requirement for idea-specific data analysis | Evidence / next action |
|---|---|
| Motivate with an initial project premise | Face-region availability and controlled insertion; describe as feasibility premises |
| Analyze original data | Full automatic screen, 117-item five-frame review, source/norm coverage and pilot-region measurements |
| Produce plots and figures | Figure 1 original-data summary and Figure 2 pilot examples |
| Discuss findings and implications for the proposal | Findings and implications above; turn into your analysis's separate discussion |
| Discuss planned analyses with primary TA before starting | Prior consultation not verified here; team must confirm |

Your analysis is ready to draft using this packet. This is one of the four analyses
required for a five-person team. The other analyses, shared introduction/discussion,
six-page ICML main-report limit, references, repository access/link and separate
AI/teammate contribution disclosure are team-level requirements, not completed
by this consolidation. Essential results must be in the main report, since the
optional appendix is not graded. Qwen inference is not a stated requirement
of this selected data-analysis category.

## Accurate AI disclosure notes

The assistant wrote and executed the downloading/screening, analysis, alignment,
compositing, figure and report-generation code; used YuNet and YOLOv8n-pose;
performed AI visual candidate/scene/task reviews; researched reference methods;
generated exploratory reference-guided face insertions; and prepared this packet.
The user directed the scope, reviewed examples and approved chunks. Do not
describe this as AI assistance with wording only or claim independent human
ratings. Record actual teammate contributions separately without inventing work.
The image-generation service did not expose a model version or random seed.

## Evidence and reproduction

consolidated_results.json stores all counts, per-frame planning measurements,
262 reconciliation checks and input hashes. data/ retains the source JSON
snapshots and previous pilot audit. code/verify_consolidation.py reproduces the
numeric reconciliation locally without models or network. The 90-frame pixel
invariant was independently verified in the previous package; this consolidation
reuses that completed audit and confirms the listed current output files exist.
It does not rerun inference or replace AI judgments with human ground truth.

The full prior archive, Full_Sequence_Face_Tests.zip, contains all 90 outputs,
raw drafts, masks, alignment matrices and the independent pixel verifier. This
smaller packet includes only the 12 full working frames used by Figure 2.

Source annotations commit: 09d8a7c53f06ed582236722ba8496c6a504cd1ab.
Source image revision: 2937df7fa96d8515417e7b5122fbf8eaeeaeee86.
Original source: https://github.com/open-social-world/EgoNormia and
https://huggingface.co/datasets/open-social-world/EgoNormia.
OMI references: https://github.com/jcpeterson/omi (CC BY-NC-SA 4.0).
Sobieszek et al.: https://doi.org/10.1111/bjop.12732, published Figure 1
references (article CC BY-NC 4.0); their generator was not executed.
Retain source terms and attribution; no broader license claim is made for this packet.
'''
(OUT/'Findings_and_Tables.md').write_text(md)
(OUT/'Figure_Captions.md').write_text('# Figure captions\n\n'+caption1+'\n\n'+caption2+'\n')
(OUT/'README.md').write_text('''# Milestone 2 consolidation

Start with **Milestone2_Consolidation.pdf** for the review packet.
**Findings_and_Tables.md** contains all tables, findings, limitations and drafting notes.
**figures/** contains the two figures in 300 dpi PNG, vector PDF and SVG.
**Figure_Captions.md** supplies captions with scope and limitations.
**consolidated_results.json** contains the numeric evidence and source hashes.
Run `python code/verify_consolidation.py` to reproduce the numeric checks;
this requires Python only. No Qwen calls or new face edits were made.
This packet precedes the final ICML analysis section and is not a submission-ready team report.
''')

# Small LaTeX tables for the later ICML write-up; the prose has not been drafted yet.
latex='\\begin{tabular}{lrr}\n\\toprule\nAssessment & Reviewed & Unreviewed \\\\\n\\midrule\nPromising region & 49 & -- \\\\\nNeeds editing test & 51 & -- \\\\\nNo suitable region identified & 17 & -- \\\\\nAwaiting visual review & -- & 1,736 \\\\\n\\bottomrule\n\\end{tabular}\n'
(OUT/'Table_Screening.tex').write_text(latex)

pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DV-B','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
W,H=A4;M=36;CW=W-2*M;INK=colors.HexColor('#172435');BLUE=colors.HexColor('#2469a0')
style=ParagraphStyle('body',fontName='DV',fontSize=9.5,leading=14,textColor=INK)
small=ParagraphStyle('small',parent=style,fontSize=8,leading=11)
PDF=ROOT/'output/pdf/Milestone2_Consolidation.pdf';PDF.parent.mkdir(exist_ok=True,parents=True)
c=canvas.Canvas(str(PDF),pagesize=A4);c.setTitle('Milestone 2: final evidence consolidation');c.setAuthor('Research review for Yash Lothe');page=0
def start(title):
 global page
 page+=1;c.setFillColor(BLUE);c.setFont('DV-B',8);c.drawString(M,H-32,'MMML / IDEA-SPECIFIC DATA ANALYSIS / REVIEW PACKET')
 c.setFillColor(INK);c.setFont('DV-B',19);c.drawString(M,H-61,title)
def para(txt,y,sty=style,width=CW,x=M):
 p=Paragraph(txt,sty);_,ph=p.wrap(width,H);p.drawOn(c,x,y-ph);return y-ph
def table(rows,widths,y):
 wrapped=[[Paragraph(str(v),small) for v in row] for row in rows]
 t=Table(wrapped,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#acbac8')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce3e9'))]));_,th=t.wrap(CW,H);t.drawOn(c,M,y-th);return y-th
def img(path,y,width=CW):
 im=Image.open(path);h=width*im.height/im.width;c.drawImage(ImageReader(im),M,y-h,width,h);return y-h
def end():
 c.setStrokeColor(colors.HexColor('#d6dfe7'));c.line(M,35,W-M,35);c.setFillColor(colors.HexColor('#536273'));c.setFont('DV',8);c.drawString(M,22,'4 October 2026 | Feasibility evidence only | No Qwen trials');c.drawRightString(W-M,22,str(page));c.showPage()

start('Verified numbers and their meaning')
y=para('The completed work supports an idea-specific feasibility analysis of Idea 3: whether EgoNormia offers suitable face regions and whether controlled insertions can preserve the scene. This packet consolidates existing evidence before writing the ICML section.',H-85)
y=table([['Evidence stage','Verified result','Scope'],['Automatic screen','1,853 items / 9,265 frames','All pinned official IDs; no failures'],['Five-frame AI review','117 items / 585 frames','Purposive selection; not human ground truth'],['Potential insertion pool','49 promising + 51 needing tests = 100','17 reviewed without a suitable region; 1,736 still unverified'],['Coverage','92 source IDs; 81 primary-norm candidates','Multi-label categories overlap; shared settings remain'],['Editing pilots','5 clips / 25 originals / 23 visible regions','Five source IDs; every pilot includes Cooperation'],['Retained outputs','75 OMI + 15 Sob-reference = 90','84 edited instances + 6 unchanged copies'],['Pixel preservation','0 changed pixels outside masks','Relative to downsampled working originals'],['Scientific validity','No validated trait-condition sets yet','Identity, realism, trait order and expression need checks']], [130,170,CW-300],y-20)
y=para('<b>Important correction:</b> 100 is the identified potential pool, not the total number of usable examples in the benchmark. The 1,736 unreviewed items cannot be classified as unusable, and purposive review cannot estimate population prevalence.',y-20)
y=para('<b>Use v2 labels:</b> the earlier one-frame audit reported 32 promising examples. Reviewing all five frames produced the current 49 promising / 51 needing tests / 17 unsuitable counts across 117 reviewed items.',y-15)
y=para('<b>Traceability:</b> reconciliation independently checked item IDs, source groups, gold labels, candidate membership, frame dimensions and output accounting. All 262 retained record-consistency checks passed. The 90-frame pixel audit was already independently completed and is reused here.',y-15)
y=para('<b>Snapshot detail:</b> the pinned annotations yield 1,075 distinct source-ID prefixes; the proposal quoted 1,077. Report the computed snapshot count and leave the discrepancy unexplained unless verified.',y-15,small)
end()

start('Original data: coverage and region size')
y=img(OUT/'figures/Figure1_Original_Data_Feasibility.png',H-83)
y=para(caption1,y-8,small)
rows=[['Correct-option norm','Promising /49','Needs test /51','Pool /100','Full /1853']]
rows += [[r['norm'],r['promising'],r['needs_test'],r['pool'],r['full']] for r in d['norm_coverage']]
y=table(rows,[180,82,82,78,CW-422],y-15)
y=para('<b>Concentration:</b> 25/100 candidates have shop-style counter/checkout labels. One apparent recurring backdrop spans 24 candidates from 22 source IDs; different IDs do not ensure different people or places. Privacy has only five candidates.',y-12,small)
end()

start('Pilot construction and remaining limits')
y=img(OUT/'figures/Figure2_Pilot_Construction_and_Limitations.png',H-82)
y=para(caption2,y-8,small)
rows=[['Item','Visible frames','Median rectangle %','Median mask support %','Main remaining issue']]
short={969:'Glasses / mouth differences',1625:'Forehead seam / occlusion',605:'Brows, eyelids and gaze',749:'More smile-like high condition',1250:'Expression / lower-face patch edge'}
rows += [[r['item'],f"{r['visible_frames']}/5",f"{r['median_region_area_percent']:.2f}",f"{r['median_mask_support_area_percent']:.2f}",short[r['item']]] for r in d['pilot_table']]
y=table(rows,[42,65,93,99,CW-299],y-12)
y=para('Working frame sizes are 640 x 360, except item 605 at 640 x 480. Planning rectangles include surrounding features; mask support includes feathering. Neither is a segmented-face measurement. The Sobieszek comparison used published references in the same compositor; their generator was not executed.',y-12,small)
end()

start('Findings ready for the write-up')
y=H-86
bullets=[
 '<b>Regions exist:</b> 100 potential examples were identified among 117 reviewed items. Unreadable original facial features do not automatically exclude a region from synthetic insertion.',
 '<b>Coverage is selective:</b> Privacy and Coordination are sparse in the pool, and recurring checkout-style scenes can dominate a convenience sample. Norm coverage does not establish task invariance.',
 '<b>Footprint varies:</b> the 23 pilot planning rectangles cover a median 3.19% of a working frame, with a 0.93-21.25% range and 46-211 pixel shortest sides. These are five-pilot measurements only.',
 '<b>Pixel preservation is achievable:</b> all outside-mask source pixels survive. Accessories, gaze, expression, identity and seams can still change inside the mask, so causal control is not established.',
 '<b>Revise the blanket blur premise:</b> four selected previews show facial detail; the food-service target is blurred. A synthetic insertion into blur does not recover a real identity.',
 '<b>No behavioral conclusion:</b> Qwen trials, text-arm trials and mitigation training were not run. No answer-flip rate, bias magnitude or successful trait ordering is claimed.'
]
for text in bullets:y=para(text,y)-15
y=para('<b>Implications for Idea 3:</b> keep constrained regions as candidates, sample across norms and reviewed source/scene groups, improve controlled insertion, then validate perceived trait ordering, identity and realism. Model trait-legibility checks should precede the main action-choice comparisons. The original plan included future Qwen requests; none was executed in this feasibility chunk.',y-6)
y=para('<b>Ready to draft:</b> use these verified results and the two figures in your analysis, followed by its own discussion. This is one of four analyses for the five-person team; it does not complete the other analyses or team-level report requirements.',y-17)
y=para('<b>Disclosure:</b> AI performed coding, automated screening, visual judgments, reference-guided image generation, compositing, analysis and document preparation. The user directed and approved the work and reviewed examples. Record actual teammate contributions; do not claim wording-only AI assistance or independent human ratings.',y-17)
y=para('<b>Still unverified administratively:</b> required prior primary-TA consultation, teammate analyses and repository access. The final team report also needs the shared introduction/discussion, six-page ICML main limit, repository link and separate AI/contribution disclosure page. Qwen inference is not a stated requirement of this data-analysis category.',y-17,small)
end();c.save();shutil.copy2(PDF,OUT/PDF.name)
print(str(PDF))
