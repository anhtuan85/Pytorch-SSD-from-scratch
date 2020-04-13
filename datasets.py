# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:09 2020

@author: NAT
"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
class VOCDataset(Dataset):
    def __init__(self, DataFolder, split):
        """
            DataFolder: folder where data files are stored
            split: split {"TRAIN", "TEST"}
        """
        self.split = str(split.upper())
        if self.split not in {"TRAIN", "TEST"}:
            print("Param split not in {TRAIN, TEST}")
            assert self.split in {"TRAIN", "TEST"}
        
        self.DataFolder = DataFolder
        
        #read data file from json file
        with open(os.path.join(DataFolder, self.split+ '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(DataFolder, self.split+ '_objects.json'), 'r') as j:
            self.objects = json.load(j)
            
        assert len(self.images) == len(self.objects)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        
        image = Image.open(self.images[i], mode= "r")
        image = image.convert("RGB")
        
        #Read objects in this image
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects["boxes"])
        labels = torch.LongTensor(objects['labels']) 
        difficulties = torch.ByteTensor(objects['difficulties'])
        
        #Apply transforms
        new_image, new_boxes, new_labels, new_difficulties = transform(image, boxes,
                                                                       labels, difficulties, self.split)
        
        return new_image, new_boxes, new_labels, new_difficulties
        
            
        