#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:21:06 2019

@author: student
"""
import torch
import numpy as np
import torch.nn as nn 
from srcnn_nopadding import SRCNN
import h5py as h5
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter
import merge_parameter_64cores as merge

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
target_data_np = np.zeros((batch_size,3,coresize,coresize,coresize))

ip = [None] * batch_num
op_u = [None] * batch_num
op_v = [None] * batch_num
op_w = [None] * batch_num
tp_u = [None] * batch_num
tp_v = [None] * batch_num
tp_w = [None] * batch_num
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.MSELoss()

#%% test loop_u
prediction_psnr = [None]*batch_num
test_psnr = [None]*batch_num
prediction_loss = [None] *batch_num
test_loss = [None]*batch_num
for i in range(batch_num):
    model = SRCNN(kernel1,kernel2,kernel3)
    model.load_state_dict(torch.load('/scratch/student/gangu/dl-turbulence-master/DL-turbulence/pure_data/9-5-5_32ch(10-02)/checkpoint_epoch_290_955_32ch.pth'))
    model.cuda(1)
    for j in range(batch_size):
        test = h5.File('/home/student/Downloads/timestep_26429/filter_width_16/u_filter_955/u_filter_'+str(j+(batch_size*i))+'.h5', 'r')
        target = h5.File('/home/student/Downloads/timestep_26429/filter_width_16/u/u_'+str(j+(batch_size*i))+'.h5', 'r') 
        
        test_data_np[j,0,:,:,:] = test['u'][:]
        target_data_np[j,0,:,:,:] = target['u'][:]
    
        test_data_np[j,1,:,:,:] = test['v'][:]
        target_data_np[j,1,:,:,:] = target['v'][:]
        
        test_data_np[j,2,:,:,:] = test['w'][:]
        target_data_np[j,2,:,:,:] = target['w'][:]
   
        
    target_data = torch.from_numpy(target_data_np)
    test_data = torch.from_numpy(test_data_np)

    target_data = target_data.float().cuda(1)   
    test_data = test_data.float().cuda(1) 
    
    prediction = model(test_data)
    prediction_loss[i] = criterion(prediction.detach(), target_data).cpu().detach().numpy()
    test_loss[i] = criterion(test_data[:,:,buffer//2:(coresize +buffer//2),buffer//2:(coresize +buffer//2),buffer//2:(coresize +buffer//2)].detach(), target_data).cpu().detach().numpy()
    

    predict = prediction.cpu().detach().numpy()
    test = test_data.cpu().detach().numpy()
    target = target_data.cpu().detach().numpy()
    
#    
    ip[i] = test[:,0,buffer//2:(coresize +buffer//2),buffer//2:(coresize +buffer//2),20]
    op_u[i] = predict[:,0,:,:,20]
    op_v[i] = predict[:,1,:,:,20]
    op_w[i] = predict[:,2,:,:,20]
    tp_u[i] = target[:,0,:,:,20]
    tp_v[i] = target[:,1,:,:,20]
    tp_w[i] = target[:,2,:,:,20]
    
    prediction_psnr[i] = 20 * math.log10(np.amax(predict)/ math.sqrt(prediction_loss[i]))
    test_psnr[i] =  20 * math.log10(np.amax(test)/math.sqrt(test_loss[i]))
    
    del target_data
    del test_data
    del model
    del prediction
    torch.cuda.empty_cache()

    print(i)
    
print('MSE',sum(prediction_loss)/len(prediction_loss))
print('MSE',sum(test_loss)/len(test_loss))
print('PSNR',sum(prediction_psnr)/len(prediction_psnr))
print('PSNR',sum(test_psnr)/len(test_psnr))

#%%merge all cores to get the original dimension before decomposition of the data
input_data = merge.merge(ip)
output_u = merge.merge(op_u)
output_v = merge.merge(op_v)
output_w = merge.merge(op_w)
target_u = merge.merge(tp_u)
target_v = merge.merge(tp_v)
target_w = merge.merge(tp_w)

#%%plot and save data
plt.figure()
plt.imshow(input_data,vmin = -3,vmax =3 )
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/input_u.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/input_u.png')

plt.figure()
plt.imshow(output_u,vmin = -3,vmax =3 )
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/output_u.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/output_u.png')
#
#
plt.figure()
plt.imshow(target_u,vmin = -3,vmax =3 )
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/target_u.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/target_u.png')
#
#%%target output off-diagonal part SGS stress tensor
target_uv = target_u *target_v
#
filter_width_train = 16
sigma_gaussian = (np.sqrt((filter_width_train**2)/12))
target_u_filter = gaussian_filter(target_u,sigma_gaussian)
target_v_filter = gaussian_filter(target_v,sigma_gaussian)
target_uv_filter = gaussian_filter(target_uv,sigma_gaussian)

tau_target = target_uv_filter - (target_u_filter*target_v_filter)
plt.figure()
plt.imshow(tau_target,vmin = -0.1,vmax =0.1 )
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_12_target.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_12_target.png')

#%%target diagonal part SGS stress tensor(with trace removal)
target_uu = target_u *target_u
target_vv = target_v *target_v
target_ww = target_w *target_w
#
filter_width_train = 16
sigma_gaussian = (np.sqrt((filter_width_train**2)/12))

target_u_filter = gaussian_filter(target_u,sigma_gaussian)
target_uu_filter = gaussian_filter(target_uu,sigma_gaussian)

target_v_filter = gaussian_filter(target_v,sigma_gaussian)
target_vv_filter = gaussian_filter(target_vv,sigma_gaussian)

target_w_filter = gaussian_filter(target_w,sigma_gaussian)
target_ww_filter = gaussian_filter(target_ww,sigma_gaussian)

tau_11 = target_uu_filter - (target_u_filter*target_u_filter)
tau_22 = target_vv_filter - (target_v_filter*target_v_filter)
tau_33 = target_ww_filter - (target_w_filter*target_w_filter)

k = (tau_11 + tau_22 + tau_33)/3

tau_11_k = tau_11 - k 
plt.figure()
plt.imshow(tau_11_k,vmin = -0.1,vmax =0.1 )
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_11_target.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_11_target.png')
#%% output off-diagonal part SGS stress tensor
output_uv = output_u *output_v
 
#
filter_width_train = 16
sigma_gaussian = (np.sqrt((filter_width_train**2)/12))

output_u_filter = gaussian_filter(output_u,sigma_gaussian)
output_v_filter = gaussian_filter(output_v,sigma_gaussian)
output_uv_filter = gaussian_filter(output_uv,sigma_gaussian)

tau_output = output_uv_filter - (output_u_filter*output_v_filter)

plt.figure()
plt.imshow(tau_output,vmin = -0.1,vmax =0.1 )
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_12_output.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_12_output.png')


#%% output diagonal part SGS stress tensor(with trace removal)
output_uu = output_u *output_u
output_vv = output_v *output_v
output_ww = output_w *output_w

#
filter_width_train = 16
sigma_gaussian = (np.sqrt((filter_width_train**2)/12))

output_u_filter = gaussian_filter(output_u,sigma_gaussian)
output_uu_filter = gaussian_filter(output_uu,sigma_gaussian)

output_v_filter = gaussian_filter(output_v,sigma_gaussian)
output_vv_filter = gaussian_filter(output_vv,sigma_gaussian)

output_w_filter = gaussian_filter(output_w,sigma_gaussian)
output_ww_filter = gaussian_filter(output_ww,sigma_gaussian)

tau_output_11 = output_uu_filter - (output_u_filter*output_u_filter)
tau_output_22 = output_vv_filter - (output_v_filter*output_v_filter)
tau_output_33 = output_ww_filter - (output_w_filter*output_w_filter)

k = (tau_output_11 + tau_output_22 + tau_output_33)/3
output_tau_11_k = tau_output_11 - k 

plt.figure()
plt.imshow(output_tau_11_k,vmin = -0.1,vmax =0.1)
plt.colorbar()
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_11_output.eps', format='eps')
plt.savefig('/home/student/Desktop/results/975_32ch_240/tau_11_output.png')
#%% calculate psnr and correlation coefficient of the reconstructed and original SGS stress tensor(off-diagonal)
target = torch.from_numpy(tau_target)
output = torch.from_numpy(tau_output)

mse = criterion(output,target)
psnr = 20 * math.log10((np.amax(tau_output))/ math.sqrt(mse))

print(mse)
print(psnr)

c = np.corrcoef(tau_target.flatten(),tau_output.flatten())
print(c)
#%%calculate psnr and correlation coefficient of the reconstructed and original SGS stress tensor(diagonal)
target_11 = torch.from_numpy(tau_11_k)
output_11 = torch.from_numpy(output_tau_11_k)
 
mse_11 = criterion(output_11,target_11)
psnr_11 = 20 * math.log10((np.amax(output_tau_11_k))/ math.sqrt(mse_11))

print(mse_11)
print(psnr_11)

c_11 = np.corrcoef(tau_11_k.flatten(),output_tau_11_k.flatten())
print(c_11)

