# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:47:25 2020

@author: NAT
"""
import sys
sys.path.append("./model/")
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from utils import *
from torchvision import transforms
import torch
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--trained_model", default= "model_state_ssd300.pth.tar", type= str,
                help = "Trained state_dict file path to open")
ap.add_argument("--input", type= str, help = "Path to the input video")
ap.add_argument("--output", type= str, help= "Path to the save video")
ap.add_argument("--min_score", default= 0.4, type= float, help = "Min score for NMS")
ap.add_argument("--max_overlap", default= 0.45, type= float, help = "Max overlap for NMS")
ap.add_argument("--top_k", default= 200, type= int, help = "Top k for NMS")
ap.add_argument("--save_fps", default = 24, type= int, help = "FPS for save output")
args = ap.parse_args()

img_path = args.input
out_path = args.output
trained_model = torch.load(args.trained_model)
ouput_path = args.output
start_epoch = trained_model["epoch"] + 1
print('\nLoaded model trained with epoch %d.\n' % start_epoch)
model = trained_model['model']
model = model.to(device)
model.eval()

#Init video stream, writer to out video
vs = cv2.VideoCapture(img_path)
writer = None
(W, H) = (None, None)

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

font = ImageFont.truetype("arial.ttf", 15)

def detect(original_image, min_score, max_overlap, top_k, suppress = None):
    image = normalize(to_tensor(resize(original_image)))
    
    image = image.to(device)
    
    locs_pred, cls_pred = model(image.unsqueeze(0))
    
    detect_boxes, detect_labels, detect_scores = model.detect(locs_pred, cls_pred, 
                                                              min_score, max_overlap, top_k)
    
    detect_boxes = detect_boxes[0].to('cpu')
    
    original_dims = torch.FloatTensor(
            [W, H, W, H]).unsqueeze(0)
    
    detect_boxes = detect_boxes * original_dims
    
    detect_labels = [rev_label_map[l] for l in detect_labels[0].to('cpu').tolist()]
    
    if detect_labels == ["background"]:
        return original_image
    
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    #font = ImageFont.truetype("arial.ttf", 15)
    
    for i in range(detect_boxes.size(0)):
        if suppress is not None:
            if detect_labels[i] in suppress:
                continue
    
        box_location = detect_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[detect_labels[i]])
        
        draw.rectangle(xy=[l + 1. for l in box_location], outline= 
                       label_color_map[detect_labels[i]])
        
        text_size = font.getsize(detect_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[detect_labels[i]])
        draw.text(xy=text_location, text=detect_labels[i].upper(), fill='white',font=font)

    return annotated_image

if __name__ == "__main__":
    while True:
        (grabbed, frame) = vs.read()
        
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        start = time.time()
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Convert image format cv2 to pil
        pil_img = Image.fromarray(converted)
        annotated_image = detect(pil_img, args.min_score, args.max_overlap, args.top_k)
        
        #Convert image format pil to cv2
        cv_annotated_image = np.array(annotated_image)
        cv_annotated_image = cv_annotated_image[:, :, ::-1].copy()
        
        if writer is None:
            writer = cv2.VideoWriter(out_path, 0x7634706d, args.save_fps ,(W, H), True)
        
        writer.write(cv_annotated_image)
        print("FPS: {0:.2f}".format(1/(time.time() - start)))

    print("Done! ^_^")
    writer.release()
    vs.release()