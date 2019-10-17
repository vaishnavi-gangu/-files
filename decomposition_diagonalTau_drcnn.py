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

#%%
#load training data
filename1='/home/student/Documents/Gangu_project/data/u_train.h5'
filename2='/home/student/Documents/Gangu_project/data/v_train.h5'
filename3='/home/student/Documents/Gangu_project/data/w_train.h5'

#load test data
#filename1='/home/student/Documents/Gangu_project/data/u_test.h5'
#filename2='/home/student/Documents/Gangu_project/data/v_test.h5'
#filename3='/home/student/Documents/Gangu_project/data/w_test.h5'

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

#%%save data
#training data
for i in range(0,num_of_filtered_velocity_files):
    print("creating training data: ",i)
    with h5.File('/home/student/Documents/Gangu_project/data/u_filter_train/u_filter_'+str(i)+'.h5', 'w') as hf:
        hf.create_dataset('u', data = u_cores[i])
        hf.create_dataset('v', data = v_cores[i])
        hf.create_dataset('w', data = w_cores[i])
        
#test data
#for i in range(0,num_of_filtered_velocity_files):
#    print("creating training data: ",i)
#    with h5.File('/home/student/Documents/Gangu_project/data/u_filter_test/u_filter_'+str(i)+'.h5', 'w') as hf:
#        hf.create_dataset('u', data = u_cores[i])
#        hf.create_dataset('v', data = v_cores[i])
#        hf.create_dataset('w', data = w_cores[i])

#%%TARGET DATA
#load training data
filename1='/home/student/Documents/Gangu_project/data/u_train.h5'
filename2='/home/student/Documents/Gangu_project/data/v_train.h5'
filename3='/home/student/Documents/Gangu_project/data/w_train.h5'

#load test data
#filename1='/home/student/Documents/Gangu_project/data/u_test.h5'
#filename2='/home/student/Documents/Gangu_project/data/v_test.h5'
#filename3='/home/student/Documents/Gangu_project/data/w_test.h5'

f = h5.File(filename1, 'r')
U = f['u'][:]
f = h5.File(filename2, 'r')
V = f['v'][:]
f = h5.File(filename3, 'r')
W = f['w'][:]
#%%product of velocities
UV = U * U
VW = V * V
WU = W * W
#%%product of filtered velocities
U = U_filter * U_filter
V = V_filter * V_filter
W = W_filter * W_filter

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
W = WU- W

k = (U+V+W)/3
#%%
U = U - k
V = V - k
W = W - k
#%%
ix = index.index(data_size,data_size,data_size,coresize,kernal_width_total)

#%%
T11_target = core.allcore(U,ix,coresize,kernal_width_total)
T22_target = core.allcore(V,ix,coresize,kernal_width_total)
T33_target = core.allcore(W,ix,coresize,kernal_width_total)

#%%
buffer = int((sum(kernal_width_total)-len(kernal_width_total))/2)

target_T11 = [None] * len(T11_target)
target_T22 = [None] * len(T22_target)
target_T33 = [None] * len(T33_target)

start = buffer
end = coresize+buffer

for i in range(len(T11_target)):
    target_T11[i] = T11_target[i][start:end,start:end,start:end]
    target_T22[i] = T22_target[i][start:end,start:end,start:end]
    target_T33[i] = T33_target[i][start:end,start:end,start:end]
    
#%%\
#save training data
num_of_target_velocity_files = len(T11_target)
for j in range(0,num_of_target_velocity_files):
    print("creating target data: ",j)
    with h5.File('/home/student/Documents/Gangu_project/data/train_diagonal_tau/tau_ii'+str(j)+'.h5', 'a') as hf:
        hf.create_dataset('T11', data = T11_target[j])
        hf.create_dataset('T22', data = T22_target[j])
        hf.create_dataset('T33', data = T33_target[j])
        
#save test data
#num_of_target_velocity_files = len(T11_target)
#for j in range(0,num_of_target_velocity_files):
#    print("creating target data: ",j)
#    with h5.File('/home/student/Documents/Gangu_project/data/test_diagonal_tau/tau_ii'+str(j)+'.h5', 'a') as hf:
#        hf.create_dataset('T11', data = T11_target[j])
#        hf.create_dataset('T22', data = T22_target[j])
#        hf.create_dataset('T33', data = T33_target[j])

