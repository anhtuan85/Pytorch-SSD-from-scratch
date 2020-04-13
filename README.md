# Pytorch Implementation of Single Shot MultiBox Detector (SSD)

This repository implements SSD300 with VGG16 backbone Architecture: [paper](https://arxiv.org/abs/1512.02325).

## Installation

* Install [Pytorch](https://pytorch.org/)
* Clone this repository (only support Python 3+)
* Download VOC dataset (currently support [VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/))

## Demo
Download pretrained model for VOC2007 dataset: [Google Drive](https://drive.google.com/file/d/1-4-k0vQD5nc_oU07J3jjVtek7R_UdPVk/view?usp=sharing)

Predict image, run `detect.py`:
```
python detect.py --trained_model path/to/pretrained/model --input path/to/the/image --output path/to/save/image
```

With video, run `video_demo.py`:
```
python video_demo.py --trained_model path/to/pretrained/model --input path/to/the/video --output path/to/save/video
```

## Training
Before training, run `create_json_data.py` to create json data from VOC downloaded data.
To train SSD run file `train.py`:
```
python train.py --dataset_root path/to/save/JSON/train/dataset --batch_size ... --lr ...
```
Default, i used SGD optimizer (same as the paper), batch size 8 (in the paper, batch size is 32), with an initial learning rate of 1e−3, 0.9 momentum, 0.0005 weight decay. I trained SSD300 with 145000 iterations, decay learning rate after 96500 iterations and 120000 iterations. 

## Evaluation
