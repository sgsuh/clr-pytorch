#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:38:21 2018

@author: seukgyo
"""

import torch.nn as nn

class CIFAR10_QUICK(nn.Module):
    def __init__(self):
        super(CIFAR10_QUICK, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(3, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(3, 2)
        
        self.ip1 = nn.Linear(576, 64)
        self.ip2 = nn.Linear(64, 10)
        
    def forward(self, img):
        out = self.conv1(img)
        out = self.pool1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        
        out = out.view(-1, 576)
        
        out = self.ip1(out)
        out = self.ip2(out)
        
        return out