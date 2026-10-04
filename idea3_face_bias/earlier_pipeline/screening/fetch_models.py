import hashlib,json,urllib.request
from pathlib import Path
r=Path(__file__).parent
m=json.loads((r/'data/method_metadata.json').read_text())
urls={'yunet.onnx':'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx','yolov8n-pose.pt':m['model_sources']['yolov8n-pose']}
# YuNet is Git LFS: use the media endpoint.
urls['yunet.onnx']='https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
(r/'models').mkdir(exist_ok=True)
for name,url in urls.items():
 p=r/'models'/name
 urllib.request.urlretrieve(url,p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==m['model_sha256'][name], 'Model bytes changed: '+name
 print('Verified',name)
