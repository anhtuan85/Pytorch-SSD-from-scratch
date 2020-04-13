# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:50:55 2020

@author: NAT
"""
import sys
sys.path.append("./model/")
from utils import *
from datasets import VOCDataset
from tqdm import tqdm
import argparse
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ap = argparse.ArgumentParser()
ap.add_argument("--data_path", default= "./JSONdata/", help= "Path to the data folder")
ap.add_argument("--batch_size", default= 8, type = int,  help = "Batch size for evaluating")
ap.add_argument("--num_workers", default= 4, type= int, help= "Number of workers")
ap.add_argument("--trained_model", default= "model_state_ssd300.pth.tar", type= str, 
                help = "Trained state_dict file path to open")
ap.add_argument("--min_score", default= 0.01, type= float, help = "Min score for NMS")
ap.add_argument("--max_overlap", default= 0.45, type= float, help = "Max overlap for NMS")
ap.add_argument("--top_k", default= 200, type= int, help = "Top k for NMS")
args = ap.parse_args()

batch_size = args.batch_size
workers = args.num_workers
data_folder = args.data_path
trained_model = torch.load(args.trained_model)
model = trained_model["model"]
model = model.to(device)

#Set eval model
model.eval()

#Load test dataset
test_dataset = VOCDataset(data_folder, split= "TEST")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, 
                                          shuffle= False, collate_fn = combine, 
                                          num_workers= workers, pin_memory= True)

def evaluate(model, test_loader):
    '''
        Evaluate model by caculate mAP
        model: model SSD
        test_loader: Dataloader for test data
        
        Out: mAP for test data
    '''
    
    model.eval()
    
    detect_boxes = []
    detect_labels = []
    detect_scores = []
    t_boxes = []
    t_labels = []
    t_difficulties = []
    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc= "Evaluating")):
            images = images.to(device)
            
            locs_pred, cls_pred = model(images)
            detect_boxes_batch, detect_labels_batch, detect_score_batch = model.detect(locs_pred, cls_pred,
                                                                                       min_score= args.min_score,
                                                                                       max_overlap = args.max_overlap,
                                                                                       top_k = args.top_k)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]
            
            detect_boxes.extend(detect_boxes_batch)
            detect_labels.extend(detect_labels_batch)
            detect_scores.extend(detect_score_batch)
            t_boxes.extend(boxes)
            t_labels.extend(labels)
            t_difficulties.extend(difficulties)
        
        APs, mAP = calculate_mAP(detect_boxes, detect_labels, detect_scores, t_boxes, t_labels, t_difficulties)
        
    print(APs)
    print("Mean Average Precision (mAP): %.3f" %mAP)

if __name__ == '__main__':
    evaluate(model, test_loader)
    
