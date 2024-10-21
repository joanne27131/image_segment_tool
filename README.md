# image_segment_tool
A simple tool by using HQ-SAM to get the segment images (remove background) and save.
## Requirements
- python>=3.8
- PyQt5
- torch==1.13.0
- torchvision==0.14.0
- sam-hq
- yolov9
##### Please refer the installation of [sam-hq](https://github.com/SysCV/sam-hq?tab=readme-ov-file)
```
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx timm
```
```
# under your working directory
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install -e .
export PYTHONPATH=$(pwd)
```
##### Please refer the installation of [YOLOv9](https://github.com/WongKinYiu/yolov9)
## Run Tool
```
python app.py
```
## Demo
![image](https://github.com/joanne27131/image_segment_tool/blob/main/demo%20img/demo.png)
