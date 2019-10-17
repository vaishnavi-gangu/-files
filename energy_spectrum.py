#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:47:12 2019

@author: student
"""

import numpy as np
import time as tt
import matplotlib.pyplot as plt
import h5py as h5

filename1='/home/student/Downloads/timestep_26429/u_26429.h5'
filename2='/home/student/Downloads/timestep_26429/v_26429.h5'
filename3='/home/student/Downloads/timestep_26429/w_26429.h5'


f = h5.File(filename1, 'r')
U = f['u'][:]
f = h5.File(filename2, 'r')
V = f['v'][:]
f = h5.File(filename3, 'r')
W = f['w'][:]

skip=2
dim=int(round(1024/skip))
print(dim)
L=2*np.pi

start = tt.time()
uu_fft=np.fft.fftn(U[::skip,::skip,::skip])
vv_fft=np.fft.fftn(V[::skip,::skip,::skip])
ww_fft=np.fft.fftn(W[::skip,::skip,::skip])
print(tt.time() - start)

start = tt.time()
uu_fft=(np.abs(uu_fft)/dim**3)**2
vv_fft=(np.abs(vv_fft)/dim**3)**2
ww_fft=(np.abs(ww_fft)/dim**3)**2
print(tt.time() - start)

k_end=int(dim/2)
rx=np.array(range(dim))-dim/2+1
rx=np.roll(rx,int(dim/2)+1)

start = tt.time()
r=np.zeros((rx.shape[0],rx.shape[0],rx.shape[0]))
for i in range(rx.shape[0]):
    for j in range(rx.shape[0]):
            r[i,j,:]=rx[i]**2+rx[j]**2+rx[:]**2
r=np.sqrt(r)
print(tt.time() - start)

dx=2*np.pi/L
k=(np.array(range(k_end))+1)*dx

start = tt.time()
bins=np.zeros((k.shape[0]+1))
for N in range(k_end):
    if N==0:
        bins[N]=0
    else:
        bins[N]=(k[N]+k[N-1])/2    
bins[-1]=k[-1]

inds = np.digitize(r*dx, bins, right=True)
spectrum=np.zeros((k.shape[0]))
bin_counter=np.zeros((k.shape[0]))

for N in range(k_end):
    spectrum[N]=np.sum(uu_fft[inds==N+1])+np.sum(vv_fft[inds==N+1])+np.sum(ww_fft[inds==N+1])
    bin_counter[N]=np.count_nonzero(inds==N+1)

spectrum=spectrum*2*np.pi*(k**2)/(bin_counter*dx**3)
print(tt.time() - start)

#%% plot the energy spectrum
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(k,spectrum)
ax.plot(k,16e-1*k**(-5/3))
ax.set_xscale('log')
ax.set_yscale('log')