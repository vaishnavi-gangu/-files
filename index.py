#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:42:38 2019

@author: student
"""

def index(x,y,z,coresize,kernel_width_total):  
    counter = 0
    buffer = int((sum(kernel_width_total)-len(kernel_width_total))/2)
    print(buffer)
    num_block = (z - (buffer*2))//coresize
    print(num_block)
    index = []
    for k in range(0,num_block):
        for j in range(0,num_block):
            for i in range(0,num_block):
                core_start =  [i*coresize +buffer,j*coresize+buffer,k*coresize+buffer]
                print (core_start)
                index.append(core_start)
                counter += 1
                
                
    print('The number of indices are')
    print(counter)
    return index                                                                                
    
    


    

      
            
          
