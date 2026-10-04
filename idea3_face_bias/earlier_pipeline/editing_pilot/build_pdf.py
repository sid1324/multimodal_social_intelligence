from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

ROOT=Path('/workspace/scratch/8eb17a085ce3');OUT=ROOT/'output/analysis/Full_Sequence_Face_Tests'
PDF=ROOT/'output/pdf/Full_Sequence_Test_Results.pdf';PDF.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
W,H=A4;M=36;CW=W-2*M
c=canvas.Canvas(str(PDF),pagesize=A4);c.setTitle('Full-sequence face-insertion feasibility tests');c.setAuthor('Research review prepared for Yash Lothe')
INK=colors.HexColor('#172435');MUTED=colors.HexColor('#536273');BLUE=colors.HexColor('#2469a0');BG=colors.HexColor('#f4f6f8')
style=ParagraphStyle('body',fontName='DejaVu',fontSize=10,leading=15,textColor=INK)
small=ParagraphStyle('small',parent=style,fontSize=8.4,leading=12)
page=0
def start(title,kicker='MMML MILESTONE 2 / FEASIBILITY TESTS'):
 global page
 page+=1;c.setFillColor(colors.white);c.rect(0,0,W,H,fill=1,stroke=0)
 c.setFillColor(BLUE);c.setFont('DejaVuBold',8);c.drawString(M,H-34,kicker)
 c.setFillColor(INK);c.setFont('DejaVuBold',19);c.drawString(M,H-65,title)
def para(txt,x,y,width=CW,sty=style):
 p=Paragraph(txt,sty);_,height=p.wrap(width,H);p.drawOn(c,x,y-height);return y-height
def end():
 c.setStrokeColor(colors.HexColor('#d6dfe7'));c.line(M,36,W-M,36)
 c.setFillColor(MUTED);c.setFont('DejaVu',8);c.drawString(M,23,'Research draft | 4 October 2026 | No trait validity claimed');c.drawRightString(W-M,23,str(page));c.showPage()
def imagefit(path,x,y,width,height):
 im=Image.open(path);scale=min(width/im.width,height/im.height);iw,ih=im.width*scale,im.height*scale;c.drawImage(ImageReader(im),x+(width-iw)/2,y+(height-ih)/2,iw,ih);return iw,ih

start('Why test all five frames?')
y=para('A first-frame test checks whether an edit is possible. A five-frame test checks whether it survives changing head pose, visibility and expression. These tests exposed alignment and expression failures that a single frame could miss.',M,H-92)
y-=28
c.setFillColor(BLUE);c.setFont('DejaVuBold',44);c.drawString(M,y-40,'90');c.setFillColor(INK);c.setFont('DejaVuBold',15);c.drawString(M+95,y-15,'variant frames completed');c.setFont('DejaVu',10);c.drawString(M+95,y-36,'84 edited face instances + 6 unchanged copies')
y-=90
y=para('<b>The scope:</b> all five sampled pre-action frames of five selected pilot clips. This is 25 source frames, not every frame of the original videos or all 1,853 examples.',M,y)
y=para('<b>OMI:</b> five clips, three trustworthiness-reference conditions, five frames each = 75 variant frames.',M,y-15)
y=para('<b>Additional test:</b> a published Sobieszek reference triplet on all five frames of item 969 = 15 variant frames. Their original generator was not executed.',M,y-15)
y=para('<b>What passed:</b> every decoded final image retains all working-original pixels outside its fixed mask. Frames with no visible target face are unchanged.',M,y-25)
y=para('<b>What remains unresolved:</b> donor identity, perceived low/baseline/high ordering, realism, glasses/hair seams, gaze and expression control. None of these sequences is approved as experiment-ready.',M,y-15)
y=para('A dataset reference does not automatically make a generated insertion a validated stimulus. Ratings of the reference faces do not transfer to the composites.',M,y-18)
y=para('How to review: the next six pages compare every face region. The ZIP includes an offline full-frame viewer, all source and output frames, masks, differences, exact prompts, code and an independent pixel verifier.',M,y-25)
end()

findings=json.loads((OUT/'review_findings.json').read_text())
donors={969:1340,1625:1437,605:1284,749:1284,1250:1320}
names={969:'Tabletop activity',1625:'Checkout',605:'Pole assembly',749:'Tubing',1250:'Blurred target face'}
for s in findings:
 item=s['item'];method=s['method'];additional=method=='sob_reference'
 start(f'{names[item]} / item {item}', 'ADDITIONAL PUBLISHED-REFERENCE TEST' if additional else 'OMI SYNTHETIC-DONOR TEST')
 y=H-92
 label='Sobieszek Figure 1 revised-method triplet' if additional else f'OMI synthetic donor {donors[item]} / trustworthiness levels 0, 1, 2'
 y=para(label,M,y,sty=small)
 # Reference strip with a source scene thumbnail gives both context and donor provenance.
 f=2 if item==749 else 1
 imagefit(OUT/f'item_{item}/original/frame_{f}.png',M,y-110,200,100)
 c.setFillColor(MUTED);c.setFont('DejaVu',8);c.drawString(M,y-122,f'Working original scene / frame {f}')
 for i,condition in enumerate(['low','baseline','high']):
  ref=OUT/'references'/f'sob_published_{condition}.png' if additional else OUT/'references'/f'omi_{donors[item]}_trustworthy_{i}.jpg'
  x=M+225+i*92;imagefit(ref,x,y-100,80,80);c.setFillColor(MUTED);c.setFont('DejaVu',8);c.drawCentredString(x+40,y-116,condition)
 y-=144
 y=para('<b>Inspection finding:</b> '+s['note'],M,y,sty=small)
 # All 20 region comparisons per page, including no-face source cells.
 grid=OUT/s['grid'];im=Image.open(grid);gh=CW*im.height/im.width
 top=y-16;imagefit(grid,M,top-gh,CW,gh)
 bottom=top-gh
 para('Rows: original, low, baseline, high. Columns: frames 1-5. Crops enlarged for inspection; enlargement adds no recovered detail. Labels describe the supplied reference, not confirmed ratings of these outputs.',M,bottom-10,sty=small)
 end()

start('What these tests establish')
y=H-96
y=para('<b>1. Construction is feasible.</b> All intended condition/frame files were produced. A fixed mask confines changes to a small region; source pixels outside it are copied exactly.',M,y)
y=para('<b>2. Registration is essential.</b> Several initial contact sheets moved the face. Background-feature matching outside the face rectangle corrected large tile shifts before final masking. The 84 registered patches had at least 163 background inliers; their maximum median inlier error was 1.02 working pixels. Background alignment does not prove facial alignment.',M,y-18)
y=para('<b>3. Preservation is narrower than validity.</b> An unchanged background cannot establish constant glasses, hair, eyes, expression or identity inside a mask. A conservative cheek mask protects most of the foreground occluder in checkout frame 3, but that patch contains very little usable trait information.',M,y-18)
y=para('<b>4. The reference can lose its signal.</b> Some results look close to the scene person; others acquire smile or eyelid differences. Neither appearance proves a successful transfer of the published trait manipulation.',M,y-18)
y=para('<b>Recommended next chunk:</b> a blinded stimulus-quality review with predeclared criteria for trait ordering, same identity, realism, expression/gaze stability and artifacts. Repair or exclude failures before VLM action tests. No review ratings or model outcomes were collected here.',M,y-23)
y=para('<b>Reproducibility:</b> the ZIP retains the exact prompts, donor references, raw drafts, alignment matrices, masks, source frames and integration code. Run code/verify.py to independently recheck all 90 final PNGs. The image tool exposes no version or random seed; exact image regeneration is not promised.',M,y-18)
y=para('<b>Pixel-check baseline:</b> downsampled originals at 640 x 360, except item 605 at 640 x 480. Native extracted source frames are also included. No-face copies: tubing frame 1 and checkout frame 4, each repeated in three conditions.',M,y-18)
y=para('<b>Reference attribution:</b> OMI, github.com/jcpeterson/omi, commit 53bb3c13113afc3ae67aa6ede4ac9dfe33f306af (CC BY-NC-SA 4.0). Sobieszek et al., DOI 10.1111/bjop.12732, Figure 1 revised-method triplet (article CC BY-NC 4.0). See README.md for links and limits. Only trustworthiness was tested; SFT and the Sobieszek generator were not run.',M,y-22,sty=small)
end();c.save()
shutil=None
import shutil
shutil.copy2(PDF,OUT/PDF.name)
print(str(PDF))
