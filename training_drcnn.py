#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:05:04 2019

@author: student
"""

import torch
import numpy as np
import torch.nn as nn 
from drcnn_nopadding import ResCNN
import torch.optim as optim
import h5py as h5
import matplotlib.pyplot as plt
import checkpoint as ch
from sklearn.utils import shuffle
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#%%

kernel1 = 9
kernel2 = 5 
kernel3 = 5

num_epochs = 291
srcnn = ResCNN(kernel1,kernel2,kernel3)
criterion = nn.MSELoss()
srcnn.cuda(0)
criterion = criterion.cuda(0)
optimiser = optim.Adam(srcnn.parameters(),lr=0.0001)

epoch_loss = 0

coresize = 64
buffer = (kernel1+kernel2+kernel3-3)

#%% training loop
total_size = 4068
batch_size = 18
batch_num = total_size//batch_size #150 batches 

#%%
def test(model):
    
    batch_size = 18
    batch_num = 1
    total_loss = 0
    test_data_np = np.zeros((batch_size,3,coresize + buffer,coresize + buffer,coresize + buffer))
    target_data_np = np.zeros((batch_size,6,coresize,coresize,coresize))
    for j in range(batch_num):
        counter = 0
        for i in range(total_size,total_size+batch_size):
            test = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/u_filter_955/u_filter_'+str(i+batch_size*j)+'.h5', 'r')
            target1 = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/tau_ij/tau_ij_'+str(i+batch_size*j)+'.h5', 'r')
            target2 = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/tau_ii/tau_ii_'+str(i+batch_size*j)+'.h5', 'r')
            
            test_data_np[counter,0,:,:,:] = test['u'][:]
            test_data_np[counter,1,:,:,:] = test['v'][:]
            test_data_np[counter,2,:,:,:] = test['w'][:]
            
            target_data_np[counter,0,:,:,:] = target1['T12'][:]
            target_data_np[counter,1,:,:,:] = target1['T23'][:]
            target_data_np[counter,2,:,:,:] = target1['T31'][:]
            target_data_np[counter,3,:,:,:] = target2['T11'][:]
            target_data_np[counter,4,:,:,:] = target2['T22'][:]
            target_data_np[counter,5,:,:,:] = target2['T33'][:]
            
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
        batch_list = rand_list[counter*batch_size:(counter+1)*batch_size]
        # load one batch
        train_data_np = np.zeros((batch_size,3,coresize + buffer,coresize + buffer,coresize + buffer))
        target_data_np = np.zeros((batch_size,6,coresize,coresize,coresize))
        for j,item in enumerate(batch_list):
            
            train = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/u_filter_955/u_filter_'+str(i+batch_size*j)+'.h5', 'r')
            target1 = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/tau_ij/tau_ij_'+str(i+batch_size*j)+'.h5', 'r')
            target2 = h5.File('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/timestep_0/filter_16/tau_ii/tau_ii_'+str(i+batch_size*j)+'.h5', 'r') 
            
            train_data_np[j,0,:,:,:] = train['u'][:]
            train_data_np[j,1,:,:,:] = train['v'][:]
            train_data_np[j,2,:,:,:] = train['w'][:]
            
            
            target_data_np[j,0,:,:,:] = target1['T12'][:]
            target_data_np[j,1,:,:,:] = target1['T23'][:]
            target_data_np[j,2,:,:,:] = target1['T31'][:]
            target_data_np[j,3,:,:,:] = target2['T11'][:]
            target_data_np[j,4,:,:,:] = target2['T22'][:]
            target_data_np[j,5,:,:,:] = target2['T33'][:]
   
            
        target_data = torch.from_numpy(target_data_np)
        target_data = target_data.float().cuda(0)
            
        train_data = torch.from_numpy(train_data_np)
        train_data = train_data.float().cuda(0) 

        optimiser.zero_grad()
        output = srcnn(train_data)
        #            tar = target_data[:,:,8:72,8:72,8:72]
        loss = criterion(output, target_data)
        epoch_loss += loss.data.item()
        loss.backward()
        optimiser.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i, batch_num, loss.data.item()))    
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
    plt.savefig('loss_drcnn_955.png')
    
    
