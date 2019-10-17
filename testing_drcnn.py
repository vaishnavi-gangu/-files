#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:26:41 2019

@author: student
"""
import torch
import numpy as np
import torch.nn as nn 
#from srcnn_3_6ch_nopadding import ResCNN
from srcnn_nopadding import SRCNN
import torch.optim as optim
import h5py as h5
from torch.utils import data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import checkpoint as ch
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter
import time
import torch.nn.functional as F 
import merge_parameter_64cores as merge



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.MSELoss()

#%%test data 
total_size = 4096
batch_size = 18
batch_num = total_size//batch_size

kernel1 = 9
kernel2 = 5
kernel3 = 5

coresize = 64
buffer = (kernel1+kernel2+kernel3-3)

test_data_np = np.zeros((batch_size,3,coresize+buffer,coresize+buffer,coresize+buffer))
target_data_np = np.zeros((batch_size,6,coresize,coresize,coresize))

ip = [None] * batch_num
op_tau_12 = [None] * batch_num
op_tau_23 = [None] * batch_num
op_tau_31 = [None] * batch_num
tp_tau_12 = [None] * batch_num
tp_tau_23 = [None] * batch_num
tp_tau_31 = [None] * batch_num
op_tau_11 = [None] * batch_num
op_tau_22 = [None] * batch_num
op_tau_33 = [None] * batch_num
tp_tau_11 = [None] * batch_num
tp_tau_22 = [None] * batch_num
tp_tau_33 = [None] * batch_num

model= torch.load('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/3layer_drcnn_955_ij/epoch_290_955_ij.pth')
model.cuda(0)


counter = 0

#%% test loop

prediction_psnr = [None]*batch_num
test_psnr = [None]*batch_num
prediction_loss = [None] *batch_num
test_loss = [None]*batch_num
for i in range(batch_num):
    for j in range(batch_size):
        test = h5.File('/home/student/Downloads/timestep_26429/filter_width_16/u_filter_955/u_filter_'+str(j+(batch_size*i))+'.h5', 'r')
        target1 = h5.File('/home/student/Downloads/timestep_26429/filter_width_16/tau_ij/tau_ij_'+str(j+(batch_size*i))+'.h5', 'r') 
        target2 = h5.File('/home/student/Downloads/timestep_26429/filter_width_16/tau_ii/tau_ii_'+str(j+(batch_size*i))+'.h5', 'r')
        
        test_data_np[j,0,:,:,:] = test['u'][:]
        target_data_np[j,0,:,:,:] = target1['T12'][:]
        target_data_np[j,3,:,:,:] = target2['T11'][:]
    
        test_data_np[j,1,:,:,:] = test['v'][:]
        target_data_np[j,1,:,:,:] = target1['T23'][:]
        target_data_np[j,4,:,:,:] = target2['T22'][:]
        
        test_data_np[j,2,:,:,:] = test['w'][:]
        target_data_np[j,2,:,:,:] = target1['T31'][:]
        target_data_np[j,5,:,:,:] = target2['T33'][:]
   
        
    target_data = torch.from_numpy(target_data_np)
    target_data = target_data.float().cuda(0)
        
    test_data = torch.from_numpy(test_data_np)
    test_data = test_data.float().cuda(0) 
    
    prediction = model(test_data)
    prediction_loss[i] = criterion(prediction.detach(), target_data)

    predict = prediction.cpu().detach().numpy()
    test = test_data.cpu().detach().numpy()
    target = target_data.cpu().detach().numpy()
    
    ip[i] = test[:,0,buffer//2:(coresize +buffer//2),buffer//2:(coresize +buffer//2),20]
    op_tau_12[i] = predict[:,0,:,:,20]
    op_tau_23[i] = predict[:,1,:,:,20]
    op_tau_31[i] = predict[:,2,:,:,20]
    op_tau_11[i] = predict[:,3,:,:,20]
    op_tau_22[i] = predict[:,4,:,:,20]
    op_tau_33[i] = predict[:,5,:,:,20]
    tp_tau_12[i] = target[:,0,:,:,20]
    tp_tau_23[i] = target[:,1,:,:,20]
    tp_tau_31[i] = target[:,2,:,:,20]
    tp_tau_11[i] = target[:,3,:,:,20]
    tp_tau_22[i] = target[:,4,:,:,20]
    tp_tau_33[i] = target[:,5,:,:,20]
    
    prediction_psnr[i] = 20 * math.log10(np.amax(predict)/ math.sqrt(prediction_loss[i]))
    
    print(i)
#%%    
print('MSE',sum(prediction_loss)/len(prediction_loss))
print('PSNR',sum(prediction_psnr)/len(prediction_psnr))


#%%
input_data = merge.merge(ip)
output_12 = merge.merge(op_tau_12)
output_23 = merge.merge(op_tau_23)
output_31 = merge.merge(op_tau_31)
target_12 = merge.merge(tp_tau_12)
target_23 = merge.merge(tp_tau_23)
target_31 = merge.merge(tp_tau_31)
output_11 = merge.merge(op_tau_11)
output_22 = merge.merge(op_tau_22)
output_33 = merge.merge(op_tau_33)
target_11 = merge.merge(tp_tau_11)
target_22 = merge.merge(tp_tau_22)
target_33 = merge.merge(tp_tau_33)

#%%
plt.figure()
plt.imshow(input_data,vmin = -3,vmax =3)
plt.colorbar()
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/input.eps', format='eps')
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/input.png')

#%%
plt.figure()
plt.imshow(output_31,vmin = -0.1,vmax =0.1)
plt.colorbar()
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/output_tau31_z20.eps', format='eps')
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/output_tau31_z20.png')

#%%


plt.figure
plt.imshow(target_31,vmin = -0.1,vmax =0.1)
plt.colorbar()
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/target_tau31_z20.eps', format='eps')
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/target_tau31_z20.png')


#%%

plt.figure
plt.imshow((output_31 - target_31),vmin = -0.1,vmax =0.1)
plt.colorbar()
plt.savefig('/home/student/Desktop/results/955_drcnn_Tij/difference_tau31_z20.png')
#%%
target = torch.from_numpy(target_31)
output = torch.from_numpy(output_31)

mse = criterion(output,target)
psnr = 20 * math.log10((np.amax(output_31))/ math.sqrt(mse))

print(mse)
print(psnr)

correlation_coeff = np.corrcoef(output_31.flatten(),target_31.flatten())
print(correlation_coeff)

