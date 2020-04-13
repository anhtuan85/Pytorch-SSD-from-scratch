# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:59:20 2020

@author: NAT
"""
from torch import nn
import torch
import sys
sys.path.append("../")
from utils import xy_to_cxcy, cxcy_to_xy, encode_bboxes, decode_bboxes, find_IoU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiBoxLoss(nn.Module):
    '''
        Loss function for model
        Localization loss + Confidence loss, Hard negative mining
    '''
    def __init__(self, default_boxes, threshold = 0.5, neg_pos= 3, alpha = 1.):
        super(MultiBoxLoss, self).__init__()
        self.default_boxes = default_boxes
        self.threshold = threshold
        self.neg_pos = neg_pos
        self.alpha = alpha
        
    def forward(self, locs_pred, cls_pred, boxes, labels):
        '''
            Forward propagation
            locs_pred: Pred location, a tensor of dimensions (N, 8732, 4)
            cls_pred:  Pred class scores for each of the encoded boxes, a tensor fo dimensions (N, 8732, n_classes)
            boxes: True object bouding boxes, a list of N tensors
            labels: True object labels, a list of N tensors
            
            Out: Mutilbox loss
        '''
        batch_size = locs_pred.size(0)    #N
        n_default_boxes = self.default_boxes.size(0)    #8732
        num_classes = cls_pred.size(2)    #num_classes
        
        t_locs = torch.zeros((batch_size, n_default_boxes, 4), dtype= torch.float).to(device) #(N, 8732, 4)
        t_classes = torch.zeros((batch_size, n_default_boxes), dtype= torch.long).to(device)    #(N, 8732)
        
        default_boxes_xy = cxcy_to_xy(self.default_boxes)
        for i in range(batch_size):
            n_objects= boxes[i].size(0)
        
            overlap = find_IoU(boxes[i], default_boxes_xy)     #(n_objects, 8732)
            
            #for each default box, find the object has maximum overlap
            overlap_each_default_box, object_each_default_box = overlap.max(dim= 0)    #(8732)
            
            #find default box has maximum oberlap for each object
            _, default_boxes_each_object = overlap.max(dim= 1)
            
            object_each_default_box[default_boxes_each_object] = torch.LongTensor(range(n_objects)).to(device)
            
            overlap_each_default_box[default_boxes_each_object] = 1.
            
            #Labels for each default box
            label_each_default_box = labels[i][object_each_default_box]    #(8732)
            
            label_each_default_box[overlap_each_default_box < self.threshold] = 0    #(8732)
            
            #Save
            t_classes[i] = label_each_default_box
            
            #Encode pred bboxes
            t_locs[i] = encode_bboxes(xy_to_cxcy(boxes[i][object_each_default_box]), self.default_boxes)    #(8732, 4)
            
        # Identify priors that are positive
        pos_default_boxes = t_classes != 0    #(N, 8732)
        
        #Localization loss
        #Localization loss is computed only over positive default boxes
        
        smooth_L1_loss = nn.SmoothL1Loss()
        loc_loss = smooth_L1_loss(locs_pred[pos_default_boxes], t_locs[pos_default_boxes])

        #Confidence loss
        #Apply hard negative mining
        
        #number of positive ad hard-negative default boxes per image
        n_positive = pos_default_boxes.sum(dim= 1)
        n_hard_negatives = self.neg_pos * n_positive
        
        #Find the loss for all priors
        cross_entropy_loss = nn.CrossEntropyLoss(reduce= False)
        confidence_loss_all = cross_entropy_loss(cls_pred.view(-1, num_classes), t_classes.view(-1))    #(N*8732)
        confidence_loss_all = confidence_loss_all.view(batch_size, n_default_boxes)    #(N, 8732)
        
        confidence_pos_loss = confidence_loss_all[pos_default_boxes]
        
        #Find which priors are hard-negative
        confidence_neg_loss = confidence_loss_all.clone()    #(N, 8732)
        confidence_neg_loss[pos_default_boxes] = 0.
        confidence_neg_loss, _ = confidence_neg_loss.sort(dim= 1, descending= True)
        
        hardness_ranks = torch.LongTensor(range(n_default_boxes)).unsqueeze(0).expand_as(confidence_neg_loss).to(device)  # (N, 8732)
        
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        
        confidence_hard_neg_loss = confidence_neg_loss[hard_negatives]
        
        confidence_loss = (confidence_hard_neg_loss.sum() + confidence_pos_loss.sum()) / n_positive.sum().float()
        
        return self.alpha * loc_loss + confidence_loss