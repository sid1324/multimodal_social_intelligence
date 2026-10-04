import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const root=path.resolve(process.env.FACE_TEST_PACKAGE || '.');
const work=root;
const out=root;
const [itemText,method,condition,generatedPath]=process.argv.slice(2);
const item=Number(itemText);
const pilot=JSON.parse(fs.readFileSync(path.join(work,'data/source_pilot_manifest.json'),'utf8'));
const record=pilot.records.find(r=>r.item===item);
if(!record || !generatedPath)throw new Error('Missing item or generated image');
const sceneDir=path.join(out,`item_${item}`,method,condition);
fs.mkdirSync(sceneDir,{recursive:true});
fs.mkdirSync(path.join(out,'raw_generated'),{recursive:true});
const rawSaved=path.join(out,'raw_generated',`item_${item}_${method}_${condition}.png`);
if(path.resolve(generatedPath)!==path.resolve(rawSaved))fs.copyFileSync(generatedPath,rawSaved);
const meta=await sharp(generatedPath).metadata();
const results=[];

function polygonFor(item,frame){
  if(item===1625 && frame===1)return [[.25,.12],[.52,.16],[.62,.35],[.6,.7],[.35,.92],[.16,.81],[.14,.64],[.02,.52],[.14,.4]];
  // Conservative visible right-cheek sliver; excludes the foreground hand/bag.
  if(item===1625 && frame===3)return [[.81,.28],[.98,.35],[.95,.65],[.82,.82],[.72,.8],[.77,.61],[.73,.46]];
  if(item===1250)return [[.29,.13],[.62,.12],[.85,.28],[.91,.56],[.77,.91],[.49,.98],[.23,.87],[.08,.58],[.12,.31]];
  return [[.28,.22],[.66,.2],[.83,.36],[.9,.61],[.76,.91],[.48,.97],[.2,.86],[.09,.6],[.12,.35]];
}
for(let f=1;f<=5;f++){
  const sourcePath=path.join(work,`item_${item}`,'original',`frame_${f}.png`);
  const source=await sharp(sourcePath).removeAlpha().raw().toBuffer({resolveWithObject:true});
  const width=source.info.width,height=source.info.height;
  const region=record.review_regions[f-1].review_rectangle_xyxy;
  const destination=path.join(sceneDir,`frame_${f}.png`);
  if(!region){
    fs.copyFileSync(sourcePath,destination);
    results.push({frame:f,editable:false,changed_pixels:0,outside_mask_changed_pixels:0,source:sourcePath,file:destination});
    continue;
  }
  const col=(f-1)%2,row=Math.floor((f-1)/2);
  const left=Math.round(col*meta.width/2),right=Math.round((col+1)*meta.width/2);
  const top=Math.round(row*meta.height/3),bottom=Math.round((row+1)*meta.height/3);
  const genUnaligned=await sharp(generatedPath).extract({left,top,width:right-left,height:bottom-top})
    .resize(width,height,{fit:'fill'}).removeAlpha().raw().toBuffer();
  const registration=JSON.parse(fs.readFileSync(path.join(out,'registration.json'),'utf8')).find(r=>r.item===item&&r.method===method&&r.condition===condition&&r.frame===f);
  if(!registration || registration.inliers<20)throw new Error('Missing reliable registration');
  const H=registration.source_to_generated;
  const gen=Buffer.from(source.data);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const den=H[2][0]*x+H[2][1]*y+H[2][2];
    const gx=(H[0][0]*x+H[0][1]*y+H[0][2])/den,gy=(H[1][0]*x+H[1][1]*y+H[1][2])/den;
    const ix=Math.floor(gx),iy=Math.floor(gy),ax=gx-ix,ay=gy-iy;
    if(ix<0||iy<0||ix+1>=width||iy+1>=height)continue;
    for(let k=0;k<3;k++)gen[(y*width+x)*3+k]=Math.round(
      genUnaligned[(iy*width+ix)*3+k]*(1-ax)*(1-ay)+
      genUnaligned[(iy*width+ix+1)*3+k]*ax*(1-ay)+
      genUnaligned[((iy+1)*width+ix)*3+k]*(1-ax)*ay+
      genUnaligned[((iy+1)*width+ix+1)*3+k]*ax*ay);
  }
  const [x0,y0,x1,y1]=region;
  const w=x1-x0,h=y1-y0;
  const points=polygonFor(item,f).map(([x,y])=>`${x0+x*w},${y0+y*h}`).join(' ');
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="black"/><polygon points="${points}" fill="white"/></svg>`;
  const maskRgb=await sharp(Buffer.from(svg)).blur(item===1625&&f===3?0.8:1.4).removeAlpha().raw().toBuffer();
  const composed=Buffer.from(source.data);
  let changed=0,outside=0,maskPixels=0,diffSum=0,rawOutsideSum=0,rawOutsideN=0;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const pix=y*width+x,idx=pix*3;
    const weight=maskRgb[idx]/255;
    let didChange=false;
    if(weight>0)maskPixels++;
    for(let k=0;k<3;k++){
      if(weight>0)composed[idx+k]=Math.round(source.data[idx+k]*(1-weight)+gen[idx+k]*weight);
      const d=Math.abs(composed[idx+k]-source.data[idx+k]);
      diffSum+=d;
      if(d)didChange=true;
      if(weight===0){rawOutsideSum+=Math.abs(gen[idx+k]-source.data[idx+k]);rawOutsideN++;}
    }
    if(didChange){changed++;if(weight===0)outside++;}
  }
  if(outside!==0)throw new Error('Pixels changed outside mask');
  await sharp(composed,{raw:{width,height,channels:3}}).png().toFile(destination);
  const masks=path.join(out,`item_${item}`,'masks');fs.mkdirSync(masks,{recursive:true});
  await sharp(maskRgb,{raw:{width,height,channels:3}}).png().toFile(path.join(masks,`frame_${f}.png`));
  results.push({frame:f,editable:true,region_xyxy:region,mask_points:points,mask_pixels:maskPixels,
    changed_pixels:changed,outside_mask_changed_pixels:outside,
    full_frame_mean_absolute_difference:diffSum/(width*height*3),
    raw_generation_outside_mask_mean_absolute_difference:rawOutsideSum/rawOutsideN,
    registration,source:sourcePath,file:destination});
}
fs.writeFileSync(path.join(sceneDir,'pixel_checks.json'),JSON.stringify({item,method,condition,raw_generation:rawSaved,
  generated_size:[meta.width,meta.height],working_frame_size:item===605?[640,480]:[640,360],frames:results},null,2));
console.log(JSON.stringify({item,method,condition,frames:results.length,editable:results.filter(x=>x.editable).length,
  outside_mask_changed_pixels:results.reduce((n,x)=>n+x.outside_mask_changed_pixels,0),files:results.map(x=>x.file)}));
