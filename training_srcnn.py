#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:41:33 2019

@author: student
"""

import torch
import numpy as np
import torch.nn as nn 
from srcnn_nopadding import SRCNN
import torch.optim as optim
import h5py as h5
import matplotlib.pyplot as plt
import checkpoint as ch
from sklearn.utils import shuffle
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%% define parameters
kernel1 = 9
kernel2 = 5
kernel3 = 5

coresize = 64
buffer = (kernel1+kernel2+kernel3-3)

num_epochs = 291
srcnn = SRCNN(kernel1,kernel2,kernel3)
criterion = nn.MSELoss()
srcnn.cuda(0)
criterion = criterion.cuda(0)
optimiser = optim.Adam(srcnn.parameters(),lr=0.0001)

epoch_loss = 0

total_size = 4068
batch_size = 18
batch_num = total_size//batch_size

#%% test function
def test(model):
    
    batch_size = 18
    batch_num = 1
    total_loss = 0
    test_data_np = np.zeros((batch_size,3,coresize + buffer,coresize + buffer,coresize + buffer))
    target_data_np = np.zeros((batch_size,3,coresize,coresize,coresize))
    for j in range(batch_num):
        counter = 0
        for i in range(total_size,total_size+batch_size):
            test = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/u_filter_955/u_filter_'+str(i+batch_size*j)+'.h5', 'r')
            target = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/u/u_'+str(i+batch_size*j)+'.h5', 'r')
            
            test_data_np[counter,0,:,:,:] = test['u'][:]
            target_data_np[counter,0,:,:,:] = target['u'][:]
        
            test_data_np[counter,1,:,:,:] = test['v'][:]
            target_data_np[counter,1,:,:,:] = target['v'][:]
            
            test_data_np[counter,2,:,:,:] = test['w'][:]
            target_data_np[counter,2,:,:,:] = target['w'][:]
            counter +=1
       
        target_data = torch.from_numpy(target_data_np)
        target_data = target_data.float().cuda(0)
            
        test_data = torch.from_numpy(test_data_np)
        test_data = test_data.float().cuda(0) 
    
        prediction = model(test_data)
        loss = criterion(prediction, target_data)
        total_loss += loss.data.item()
    total_loss = total_loss/batch_num
    return total_loss

#%% training loop
loss_history = []
validate_loss_history = []
for epoch in range(num_epochs):   
    t = time.time()
    counter = 0
    epoch_loss = 0
    rand_list = shuffle(list(range(total_size)))
    for i in range(batch_num):
        #shuffle data in batch
        batch_list = rand_list[counter*batch_size:(counter+1)*batch_size]
        # load one batch
        train_data_np = np.zeros((batch_size,3,coresize + buffer,coresize + buffer,coresize + buffer))
        target_data_np = np.zeros((batch_size,3,coresize,coresize,coresize))
        for j,item in enumerate(batch_list):
            
            train = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/u_filter_955/u_filter_'+str(item)+'.h5', 'r')
            target = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/u/u_'+str(item)+'.h5', 'r') 
            
            train_data_np[j,0,:,:,:] = train['u'][:]
            target_data_np[j,0,:,:,:] = target['u'][:]
        
            train_data_np[j,1,:,:,:] = train['v'][:]
            target_data_np[j,1,:,:,:] = target['v'][:]
            
            train_data_np[j,2,:,:,:] = train['w'][:]
            target_data_np[j,2,:,:,:] = target['w'][:]
   
            
        target_data = torch.from_numpy(target_data_np)
        target_data = target_data.float().cuda(0)
            
        train_data = torch.from_numpy(train_data_np)
        train_data = train_data.float().cuda(0) 

        optimiser.zero_grad()
        output = srcnn(train_data)
        loss = criterion(output, target_data)
        epoch_loss += loss.data.item()
        loss.backward()
        optimiser.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i+1, batch_num, loss.data.item()))    
        counter += 1
        
    elapsed = time.time() - t
    print(elapsed) 
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss/batch_num))
    if(epoch%10==0):
        ch.checkpoint(epoch,srcnn)
        
    validate_loss = test(srcnn)
    loss_history.append(epoch_loss/batch_num)
    validate_loss_history.append(validate_loss) 
    plt.clf()
    plt.semilogy(loss_history)
    plt.semilogy(validate_loss_history,'k--')
    plt.savefig('loss_955.png')

