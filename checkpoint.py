#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:29:26 2019

@author: student
"""
import torch

def checkpoint(num_epoch,model):
    model_out_path = "drcnn_epoch_{}_975_32ch.pth".format(num_epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
