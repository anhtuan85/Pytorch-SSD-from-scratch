# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:15:16 2020

@author: NAT
"""
from utils import parse_annotation, save_label_map
import os
import json

def create_json_data(voc_path, output_path):
    '''
        Create json file saved data
        voc_path: Path to VOC folder
        output_path: Path to save folder
    '''
    #TRain
    train_images = list()
    train_objects = list()
    
    with open(os.path.join(voc_path, "ImageSets/Main/trainval.txt")) as f:
        ids = f.read().splitlines()
    
    for id in ids:
        #Get object, image in XML file
        object_path = os.path.join(voc_path, "Annotations", id + ".xml")
        objects = parse_annotation(object_path)
        if len(objects) == 0:
            continue
        train_objects.append(objects)
        
        image_path = voc_path + "/JPEGImages/"+id+".jpg"
        train_images.append(image_path)
        
    assert len(train_objects) == len(train_images)
    
    #Save to file
    with open(os.path.join(output_path, "TRAIN_images.json"), "w") as j:
        json.dump(train_images, j)
    
    with open(os.path.join(output_path, "TRAIN_objects.json"), "w") as j:
        json.dump(train_objects, j)
    
    #Test
    test_images = list()
    test_objects = list()
    
    with open(os.path.join(voc_path, "ImageSets/Main/test.txt")) as f:
        ids = f.read().splitlines()
        
    for id in ids:
        object_path = os.path.join(voc_path, "Annotations", id + ".xml")
        objects = parse_annotation(object_path)
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        test_image_path = voc_path + "/JPEGImages/"+id+".jpg"
        test_images.append(test_image_path)
    
    assert len(test_images) == len(test_objects)
    
    #Save to file
    with open(os.path.join(output_path, "TEST_images.json"), "w") as j:
        json.dump(test_images, j)
    with open(os.path.join(output_path, "TEST_objects.json"), "w") as j:
        json.dump(test_objects, j)
        
create_json_data("./VOCdevkit/VOC2007","./JSONdata")