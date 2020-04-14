# Pytorch Implementation of Single Shot MultiBox Detector (SSD)

This repository implements SSD300 with VGG16 backbone Architecture: [paper](https://arxiv.org/abs/1512.02325).

## Installation

* Install [Pytorch](https://pytorch.org/)
* Clone this repository (only support Python 3+)
* Download VOC dataset (currently support [VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/))
* Install requirements:
```
pip install -r requirements.txt
```

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
python train.py --dataset_root path/to/saved/JSON/train/dataset --batch_size ... --lr ...
```
Default, i used SGD optimizer (same as the paper), batch size 8 (in the paper, batch size is 32), with an initial learning rate of 1e−3, 0.9 momentum, 0.0005 weight decay. I trained SSD300 with 145000 iterations, decay learning rate after 96500 iterations and 120000 iterations. 

## Evaluation
To evaluate a trained model:
```
python eval.py --trained_model path/to/pretrained/model --data_path path/to/saved/JSON/test/dataset --batch_size ...
```
Default, i set at min_score of 0.01, an NMS max_overlap of 0.45, and top_k of 200 to allow fair comparision of results with the paper.

## Performance
I only train on VOC2007trainval. The model scores 70.7 mAP on VOC2007 test.
Class-wise average precisions:
|  Class      |      AP      |
| :---------: | :----------: |
|  aeroplane  |    73.71     |
|   bicycle   |    80.97     |
|    bird     |    68.79     |
|    boat     |    60.82     |
|   bottle    |    37.73     |
|    bus      |    82.13     |
|    car	    |    81.88     |
|    cat      |    81.71     |
|   chair     |    49.05     |
|    cow      |    74.69     |
| diningtable |    67.29     |
|    dog      |    80.32     |
|    horse    |    83.71     |
|  motorbike  |    80.78     |
|   person    |    73.67     |
| pottedplant	|    40.22     |
|    sheep    |    68.47     |
|    sofa     |    75.23     |
|   train     |    83.59     |
|  tvmonitor  |    69.59     |

Some examples in folder `images`
## TODO 
I hope to complete the to-do list in the near future (Never give up!):

* [ ] Train model on VOC2012 + VOC2007 dataset (I need a more powerful GPU -_- ).
* [ ] Support for MS COCO dataset.
* [ ] Implement SSD512.
* [ ] Implement SSD with other backbone (ResNet, MobileNet,...).
* [ ] Implement Gui for object detection.
