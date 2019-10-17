#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:06:24 2019

@author: student
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
import matplotlib.pyplot as plt
from PIL import Image


class ResCNN(nn.Module):
    def __init__(self,kernel1,kernel2,kernel3):
        super(ResCNN,self).__init__()
        self.conv1 = nn.Conv3d(3,32,kernel_size=kernel1,padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(32,32,kernel_size=kernel2,padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(32,6,kernel_size=kernel3,padding=0)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out
