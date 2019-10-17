#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:59:49 2019

@author: student
"""

import numpy as np 

def enlarge(inputarray,orignal_n,filtersize,volume_of_padding):
    
    if (filtersize % 2 == 0):
        filtersize = (filtersize * volume_of_padding) + 1
    else:
        filtersize = (filtersize * volume_of_padding) - 1
    
    buffer = (filtersize-1)//2
    print(buffer)
    
    face_start = [None]*3
    face_end = [None]*3
    edge_end = [None]*6
    edge_start = [None]*6
    
    for i in range(3):
        face_start[i] = inputarray.take(indices = range(orignal_n-buffer,orignal_n), axis = i)
        face_end[i] = inputarray.take(indices = range(0,buffer), axis = i)
        
        dimension = [1,2,0]
        j = dimension[i]
        edge_end[i] = face_end[i].take(indices = range(0,buffer), axis = j)
        edge_end[i+3] = face_end[i].take(indices = range(orignal_n-buffer,orignal_n), axis = j)
        edge_start[i] = face_start[i].take(indices = range(0,buffer), axis = j)
        edge_start[i+3] = face_start[i].take(indices = range(orignal_n-buffer,orignal_n), axis = j)
    
        
    vertex_end = [None]*4
    vertex_start = [None]*4
    
    
    
    for j in range(0,6,3):
        dimension = [0,1]
        i = dimension[j-2]
        vertex_end[i] = edge_end[j].take(indices = range(0,buffer),axis = 2)
        vertex_end[i+2] =  edge_end[j].take(indices = range(orignal_n-buffer,orignal_n),axis = 2)
        vertex_start[i] = edge_start[j].take(indices = range(0,buffer),axis = 2)
        vertex_start[i+2] =  edge_start[j].take(indices = range(orignal_n-buffer,orignal_n),axis = 2)
    
    
    
    
    
    
    N = orignal_n+filtersize-1
    enlarged_data = np.zeros((N,N,N))
    
    enlarged_data[buffer:N-buffer,buffer:N-buffer,buffer:N-buffer] = inputarray
    
    enlarged_data[0:buffer,buffer:(N-buffer),buffer:(N-buffer)] = face_start[0]
    enlarged_data[(N-buffer):N,buffer:(N-buffer),buffer:(N-buffer)] = face_end[0]
    enlarged_data[buffer:(N-buffer),0:buffer,buffer:(N-buffer)] = face_start[1]
    enlarged_data[buffer:(N-buffer),(N-buffer):N,buffer:(N-buffer)] = face_end[1]
    enlarged_data[buffer:(N-buffer),buffer:(N-buffer),0:buffer] = face_start[2]
    enlarged_data[buffer:(N-buffer),buffer:(N-buffer),(N-buffer):N] = face_end[2]
    
    enlarged_data[buffer:(N-buffer),0:buffer,0:buffer] = edge_start[4]
    enlarged_data[buffer:(N-buffer),(N-buffer):N,0:buffer] = edge_end[4]
    enlarged_data[buffer:(N-buffer),0:buffer,(N-buffer):N] = edge_start[1]
    enlarged_data[buffer:(N-buffer),(N-buffer):N,(N-buffer):N] = edge_end[1]
    enlarged_data[0:buffer,buffer:(N-buffer),0:buffer]=edge_start[5]
    enlarged_data[(N-buffer):N,buffer:(N-buffer),0:buffer]=edge_start[2]
    enlarged_data[0:buffer,buffer:(N-buffer),(N-buffer):N]=edge_end[5]
    enlarged_data[(N-buffer):N,buffer:(N-buffer),(N-buffer):N]=edge_end[2]
    enlarged_data[0:buffer,0:buffer,buffer:(N-buffer)]=edge_start[3]
    enlarged_data[0:buffer,(N-buffer):N,buffer:(N-buffer)]=edge_start[0]
    enlarged_data[(N-buffer):N,0:buffer,buffer:(N-buffer)]=edge_end[3]
    enlarged_data[(N-buffer):N,(N-buffer):N,buffer:(N-buffer)]=edge_end[0]
    
    
    
    enlarged_data[0:buffer,0:buffer,0:buffer]=vertex_start[3]
    enlarged_data[(N-buffer):N,0:buffer,0:buffer]=vertex_end[3]
    enlarged_data[0:buffer,(N-buffer):N,0:buffer]=vertex_start[2]
    enlarged_data[(N-buffer):N,(N-buffer):N,0:buffer]=vertex_end[2]
    enlarged_data[0:buffer,0:buffer,(N-buffer):N]=vertex_start[1]
    enlarged_data[(N-buffer):N,0:buffer,(N-buffer):N]=vertex_end[1]
    enlarged_data[0:buffer,(N-buffer):N,(N-buffer):N]=vertex_start[0]
    enlarged_data[(N-buffer):N,(N-buffer):N,(N-buffer):N]=vertex_end[0]

    
    return enlarged_data
