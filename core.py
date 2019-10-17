#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:52:22 2019

@author: student
"""

import extract as ex

def allcore(inputarray,indexarray,coresize,kernel_width_total):
    core = []
    for j in indexarray:
        c = ex.extract(j,inputarray,coresize,kernel_width_total)
        core.append(c)
    
    return core
