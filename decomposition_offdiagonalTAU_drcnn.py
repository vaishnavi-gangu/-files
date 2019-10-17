#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:09:44 2019

@author: student
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:19:39 2019
@author: student
"""

import numpy as np
import h5py as h5
import enlarge as el
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
#%%TRAINING DATA or TESTING DATA
#padding for filtering 
U_filter = el.enlarge(U,data_size,filter_width_train,2)
V_filter = el.enlarge(V,data_size,filter_width_train,2)
W_filter = el.enlarge(W,data_size,filter_width_train,2)
#%%
data_size = round(U_filter.size**(1/3))
print((data_size))

#%%filter ground truth using GAUSSIAN filter
U_filter = gaussian_filter(U_filter,sigma_gaussian)
V_filter = gaussian_filter(V_filter,sigma_gaussian)
W_filter = gaussian_filter(W_filter,sigma_gaussian)

#%%
data_size = round(U_filter.size**(1/3))
print((data_size))

#%%remove the padding
U_filter = U_filter[padding_each_side :(data_size - padding_each_side), padding_each_side :(data_size - padding_each_side) ,padding_each_side :(data_size - padding_each_side) ]
V_filter = V_filter[padding_each_side :(data_size - padding_each_side), padding_each_side :(data_size - padding_each_side) ,padding_each_side :(data_size - padding_each_side) ]
W_filter = W_filter[padding_each_side :(data_size - padding_each_side), padding_each_side :(data_size - padding_each_side) ,padding_each_side :(data_size - padding_each_side) ]
#%%
data_size = round(U_filter.size**(1/3))
print((data_size))
#%% padding for convolution operation
U = el.enlarge(U_filter,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
V = el.enlarge(V_filter,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
W = el.enlarge(W_filter,data_size,(sum(kernal_width_total)-len(kernal_width_total)),1)
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

##%%save data
#for i in range(0,num_of_filtered_velocity_files):
#    print("creating training data: ",i)
#    with h5.File('/home/student/Downloads/timestep_26429/filter_width_16/u_filter_955/u_filter_'+str(i)+'.h5', 'w') as hf:
#        hf.create_dataset('u', data = u_cores[i])
#        hf.create_dataset('v', data = v_cores[i])
#        hf.create_dataset('w', data = w_cores[i])

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
#%%product of velocities
UV = U * V
VW = V * W
WU = W * U
#%%product of filtered velocities
U = U_filter * V_filter
V = V_filter * W_filter
W = W_filter * U_filter

#%%filtered product of velocities
UV = gaussian_filter(UV,sigma_gaussian)
VW = gaussian_filter(VW,sigma_gaussian)
WU = gaussian_filter(WU,sigma_gaussian)

#%%
data_size = round(U_filter.size**(1/3))
print((data_size))

#%% SGS stress tensor
U = UV - U
V = VW - V
W = WU - W

#%%
ix = index.index(data_size,data_size,data_size,coresize,kernal_width_total)

#%%
T12_target = core.allcore(U,ix,coresize,kernal_width_total)
T23_target = core.allcore(V,ix,coresize,kernal_width_total)
T31_target = core.allcore(W,ix,coresize,kernal_width_total)

#%%
buffer = int((sum(kernal_width_total)-len(kernal_width_total))/2)

target_T12 = [None] * len(T12_target)
target_T23 = [None] * len(T23_target)
target_T31 = [None] * len(T31_target)

start = buffer
end = coresize+buffer

for i in range(len(T12_target)):
    target_T12[i] = T12_target[i][start:end,start:end,start:end]
    target_T23[i] = T23_target[i][start:end,start:end,start:end]
    target_T31[i] = T31_target[i][start:end,start:end,start:end]
    
##%%
#num_of_target_velocity_files = len(T12_target)
#for j in range(0,num_of_target_velocity_files):
#    print("creating target data: ",j)
#    with h5.File('/home/student/Downloads/timestep_26429/filter_width_16/Tau_ij_trace/tau_ij'+str(j)+'.h5', 'a') as hf:
#        hf.create_dataset('T12', data = T12_target[j])
#        hf.create_dataset('T23', data = T23_target[j])
#        hf.create_dataset('T31', data = T31_target[j])

