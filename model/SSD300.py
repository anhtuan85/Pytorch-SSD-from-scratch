# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:30:06 2020

@author: NAT
"""
from torch import nn
import torch.nn.functional as F
import torch
import sys
sys.path.append("../")
from vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet
from L2Norm import L2Norm
import math
from utils import xy_to_cxcy, cxcy_to_xy, encode_bboxes, decode_bboxes, find_IoU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SSD300(nn.Module):
    '''
        SSD300 network, VGG16-backbone
    '''
    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        
        self.num_classes = num_classes
        self.base_model = VGG16BaseNet(pretrained= True)
        self.aux_net = AuxiliaryNet()
        self.pred_net = PredictionNet(num_classes)
        
        #Rescale factor for conv4_3, it is learned during back-prop
        self.L2Norm = L2Norm(channels= 512, scale= 20)
        
        #Prior boxes coordinate cx, cy, w, h
        self.default_boxes = self.create_default_boxes()
    def forward(self, image):
        '''
            Forward propagation
            image: images, a tensor of dimension (N, 3, 300, 300)
            
            Out: 8732 pred loc, classes for each image
        '''
        conv4_3_out, conv7_out = self.base_model(image)     #(N, 512, 38, 38), (N, 1024, 19, 19)
        
        conv4_3_out = self.L2Norm(conv4_3_out)    #(N, 512, 38, 38)
        
        conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = self.aux_net(conv7_out)
        #(N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
        
        locs_pred, cls_pred = self.pred_net(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out,
                                            conv10_2_out, conv11_2_out)    #(N, 8732, 4) #(N, 8732, classes)
        
        return locs_pred, cls_pred
        
    def create_default_boxes(self):
        '''
            Create 8732 default boxes in center-coordinate,
            a tensor of dimensions (8732, 4)
        '''
        fmap_wh = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5,
                   "conv10_2": 3, "conv11_2": 1}
        
        scales = {"conv4_3": 0.1, "conv7": 0.2, "conv8_2": 0.375,
                  "conv9_2": 0.55, "conv10_2": 0.725, "conv11_2": 0.9}
        
        aspect_ratios= {"conv4_3": [1., 2., 0.5], "conv7": [1., 2., 3., 0.5, 0.3333],
                        "conv8_2": [1., 2., 3., 0.5, 0.3333], 
                        "conv9_2": [1., 2., 3., 0.5, 0.3333],
                        "conv10_2": [1., 2., 0.5], "conv11_2": [1., 2., 0.5]}
        
        fmaps = list(fmap_wh.keys())
        
        default_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_wh[fmap]):
                for j in range(fmap_wh[fmap]):
                    cx = (j + 0.5) / fmap_wh[fmap]
                    cy = (i + 0.5) / fmap_wh[fmap]
                    
                    for ratio in aspect_ratios[fmap]:
                        default_boxes.append([cx, cy, scales[fmap]* math.sqrt(ratio), 
                                              scales[fmap]/math.sqrt(ratio)]) #(cx, cy, w, h)
                        
                        if ratio == 1:
                            try:
                                add_scale = math.sqrt(scales[fmap]*scales[fmaps[k+1]])
                            except IndexError:
                                #for the last feature map
                                add_scale = 1.
                            default_boxes.append([cx, cy, add_scale, add_scale])
        
        default_boxes = torch.FloatTensor(default_boxes).to(device) #(8732, 4)
        default_boxes.clamp_(0, 1)
        assert default_boxes.size(0) == 8732
        assert default_boxes.size(1) == 4
        return default_boxes
    
    def detect(self, locs_pred, cls_pred, min_score, max_overlap, top_k):
        '''
            Detect objects, perform NMS on boxes that are above a minimum threshold.
            locs_pred: Pred location, a tensor of dimensions (N, 8732, 4)
            cls_pred: Pred class scores for each of the encoded boxes, a tensor fo dimensions (N, 8732, n_classes)
            min_score: min threshold 
            max_overlap: maximum overlap two boxes can have 
            top_k: if there are lot of resulting detection across all classes, keep only the top 'k'
            
            Out: detection list: boxes, labels, score
        '''
        batch_size = locs_pred.size(0)    #N
        n_default_boxes = self.default_boxes.size(0)    #8732
        cls_pred = F.softmax(cls_pred, dim= 2)    #(N, 8732, n_classes)
        assert n_default_boxes == locs_pred.size(1) == cls_pred.size(1)
        
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []
        
        for i in range(batch_size):
            #Decode object
            decoded_locs = cxcy_to_xy(decode_bboxes(locs_pred[i], self.default_boxes)) #(8732, 4)
            
            image_boxes = []
            image_labels = []
            image_scores = []
            
            max_scores, best_label = cls_pred[i].max(dim= 1)    #(8732)
            
            #Check for each class
            for c in range(1, self.num_classes):
                class_scores = cls_pred[i][:, c]    #8732
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                
                if n_above_min_score == 0:
                    continue
                
                class_scores = class_scores[score_above_min_score]    # <=8732
                class_decoded_locs = decoded_locs[score_above_min_score] # <=8732
                
                #Sort pred boxes and socores by scores
                class_scores, sort_id = class_scores.sort(dim= 0, descending= True)
                class_decoded_locs = class_decoded_locs[sort_id]
                
                #Find overlap between pred locs
                overlap = find_IoU(class_decoded_locs, class_decoded_locs)
                
                #Apply NMS
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)
                
                for box_id in range(class_decoded_locs.size(0)):
                    if suppress[box_id] == 1:
                        continue
                    condition = overlap[box_id] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(device)
                    suppress = torch.max(suppress, condition)
                    
                    suppress[box_id] = 0
                
                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])
            
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))
            
            #Concat into single tensors
            image_boxes = torch.cat(image_boxes, dim= 0)    #(n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)
            
            #Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_index = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_index][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_index][:top_k]  # (top_k)
            
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
            
        return all_images_boxes, all_images_labels, all_images_scores        