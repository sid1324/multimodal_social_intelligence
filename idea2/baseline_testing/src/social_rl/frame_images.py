"""Encode released chronological frames as images, without invented timestamps."""
from PIL import Image

def ordered_image_inputs(policy,prompt,paths):
 images=[]
 for path in paths:
  with Image.open(path) as f:im=f.convert('RGB')
  im.thumbnail((256,256),Image.Resampling.LANCZOS);images.append(im)
 content=[{'type':'image'} for _ in images]+[{'type':'text','text':'The supplied images are video frames in chronological order.\n'+prompt}]
 text=policy.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 batch=policy.processor(text=[text],images=images,return_tensors='pt',size={'shortest_edge':32*32,'longest_edge':256*256})
 return batch.to(policy.device)
