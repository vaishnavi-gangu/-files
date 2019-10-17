#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:52:47 2019

@author: student
"""

def extract(index,data,coresize,kernel_width_total):
    buffer = int((sum(kernel_width_total)-len(kernel_width_total))/2)
    out = data[index[0]-buffer :index[0]+coresize+buffer,index[1]-buffer :index[1]+coresize+buffer,index[2]-buffer :index[2]+coresize+buffer]
    return out
