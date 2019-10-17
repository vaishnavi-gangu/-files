#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:54:35 2019

@author: student
"""


import numpy as np
import h5py as h5
import enlarge as el
import matplotlib.pyplot as plt
import index 
import core as core
from scipy.ndimage import gaussian_filter

#%%load data 
filename1='/home/student/Downloads/timestep_26429/u_26429.h5'
filename2='/home/student/Downloads/timestep_26429/v_26429.h5'
filename3='/home/student/Downloads/timestep_26429/w_26429.h5'


f = h5.File(filename1, 'r')
U = f['u'][:]
f = h5.File(filename2, 'r')
V = f['v'][:]
f = h5.File(filename3, 'r')
W = f['w'][:]

#%%
data_size = round(U.size**(1/3))
print((data_size))

#%% specify value of parameters 
filter_width_train = 16
sigma_gaussian = (np.sqrt((filter_width_train**2)/12))
coresize = 64
kernal_width_total = [9,5,5]
padding_each_side = filter_width_train#only for even numbers
filter_width_target = 1 ##always 1

#%%TRAINING DATA or TEST DATA
# padding for filering  
U = el.enlarge(U,data_size,filter_width_train,2)
V = el.enlarge(V,data_size,filter_width_train,2)
W = el.enlarge(W,data_size,filter_width_train,2)
#%%
data_size = round(U.size**(1/3))
print((data_size))

#%%filter DNS data using GAUSSIAN filter
U = gaussian_filter(U,sigma_gaussian)
V = gaussian_filter(V,sigma_gaussian)
W = gaussian_filter(W,sigma_gaussian)

#%%
data_size = round(U.size**(1/3))
print((data_size))

#%%remove padding 
U = U[padding_each_side :(data_size - padding_each_side), padding_each_side :(data_size - padding_each_side) ,padding_each_side :(data_size - padding_each_side) ]
V = V[padding_each_side :(data_size - padding_each_side), padding_each_side :(data_size - padding_each_side) ,padding_each_side :(data_size - padding_each_side) ]
W = W[padding_each_side :(data_size - padding_each_side), padding_each_side :(data_size - padding_each_side) ,padding_each_side :(data_size - padding_each_side) ]
#%%
data_size = round(U.size**(1/3))
print((data_size))

#%% padding for convolution operation
U = el.enlarge(U,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
V = el.enlarge(V,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
W = el.enlarge(W,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
#%%
data_size = round(U.size**(1/3))
print((data_size))
#%% extract starting indices of the core cubes
ix = index.index(data_size,data_size,data_size,coresize,kernal_width_total)

#%%extract cores of filtered data
u_cores = core.allcore(U,ix,coresize,kernal_width_total)
v_cores = core.allcore(V,ix,coresize,kernal_width_total)
w_cores = core.allcore(W,ix,coresize,kernal_width_total)
#
#%%
num_of_filtered_velocity_files = len(u_cores)

#%%save data
for i in range(0,num_of_filtered_velocity_files):
    print("creating training data: ",i)
    with h5.File('/home/student/Downloads/timestep_26429/filter_width_16/u_filter_955/u_filter_'+str(i)+'.h5', 'w') as hf:
        hf.create_dataset('u', data = u_cores[i])
        hf.create_dataset('v', data = v_cores[i])
        hf.create_dataset('w', data = w_cores[i])


#%%TARGET DATA
#load data
filename1='/home/student/Downloads/timestep_26429/u_26429.h5'
filename2='/home/student/Downloads/timestep_26429/v_26429.h5'
filename3='/home/student/Downloads/timestep_26429/w_26429.h5'


f = h5.File(filename1, 'r')
U = f['u'][:]
f = h5.File(filename2, 'r')
V = f['v'][:]
f = h5.File(filename3, 'r')
W = f['w'][:]

#%%
data_size = round(U.size**(1/3))
print((data_size))

#%% specify value of parameters 
filter_width_target = 1
sigma_gaussian = (np.sqrt((filter_width_target**2)/12))
coresize = 64
kernal_width_total = [9,5,5]
padding_each_side = filter_width_target#only for even numbers

#%%padding for convolution operation
U = el.enlarge(U,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
V = el.enlarge(V,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
W = el.enlarge(W,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
#%%
data_size = round(U.size**(1/3))
print((data_size))

#%% extract starting indices of the core cubes
ix = index.index(data_size,data_size,data_size,coresize,kernal_width_total)

#%%extract cores of target data
U_target = core.allcore(U,ix,coresize,kernal_width_total)
V_target = core.allcore(V,ix,coresize,kernal_width_total)
W_target = core.allcore(W,ix,coresize,kernal_width_total)

#%% remove padding of each core 
buffer = int((sum(kernal_width_total)-len(kernal_width_total))/2)

target_u = [None] * len(U_target)
target_v = [None] * len(V_target)
target_w = [None] * len(W_target)

start = buffer
end = coresize+buffer

for i in range(len(target_u)):
    target_u[i] = U_target[i][start:end,start:end,start:end]
    target_v[i] = V_target[i][start:end,start:end,start:end]
    target_w[i] = W_target[i][start:end,start:end,start:end]
    
#%%save data
num_of_target_velocity_files = len(target_u)
for j in range(0,num_of_target_velocity_files):
    print("creating target data: ",j)
    with h5.File('/home/student/Downloads/timestep_26429/filter_width_16/u_975/u_'+str(j)+'.h5', 'w') as hf:
        hf.create_dataset('u', data = target_u[j])
        hf.create_dataset('v', data = target_v[j])
        hf.create_dataset('w', data = target_w[j])
#%%
plt.figure()
plt.imshow(u_cores[110][8,8:136,8:136])
plt.colorbar()

plt.figure()
plt.imshow(target_u[110][0,:,:])
plt.colorbar()


