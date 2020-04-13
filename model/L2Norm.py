# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:40:39 2020

@author: NAT
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, channels, scale):
        super(L2Norm, self).__init__()
        
        self.channels = channels
        self.scale = scale
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, channels, 1, 1))
        
        self.reset_params()
        
    def reset_params(self):
        init.constant_(self.rescale_factors, self.scale)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = x / norm
        out = x * self.rescale_factors
        return out
        
        