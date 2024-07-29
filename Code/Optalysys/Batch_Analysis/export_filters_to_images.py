# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:38:42 2021

@author: eers500
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui

p = gui.fileopenbox()
lut = f.videoImport(p, 0)

# p = 'F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\'
# lut = f.videoImport(p+'\\LUT.avi', 0)

#%% Convert to binary
ni, nj, nk = lut.shape
lut_binary = np.zeros_like(lut)
means = np.mean(lut, axis=(0, 1))

for k in range(nk):
    lut_binary[lut >= means[k]] = 255

#%% Pad array
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
#
target_size = 180
# target_size = 500

padded_array = np.empty((target_size, target_size, nk))
pi, pj, pk = padded_array.shape

for k in range(pk):
    # BB[:, :, k] = np.pad(B[:, :, k], int((1024-110)/2))
    padded_array[:, :, k] = np.pad(lut_binary[:, :, k], int((pi-ni)/2))

#%% Create filters
FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
# IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

# FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))

# ft = np.empty_like(padded_array, dtype='complex128')
# filt = np.zeros_like(padded_array)
# for k in range(pk):
#     ft[:, :, k] = FT(padded_array[:, :, k])
#     ff = np.abs(ft[:, :, k])
#     ffz = np.zeros_like(ff)
#     ffz[ff >= ff.mean()] = 255
#     filt[:, :, k] = ffz
    
# plt.imshow(np.real(ft[:,:,0]), cmap='jet')    

binaryPhaseFilter = np.empty_like(padded_array)    
phaseFilter = np.empty_like(padded_array)
for k in range(pk):
    refFT = FT(padded_array[:, :, k])
    phaseFilter[:, :, k] = -np.angle(refFT)
    binaryPhaseFilter[:, :, k] = 255*(phaseFilter[:, :, k]>0)


f.exportAVI('phase_filters.avi', binaryPhaseFilter.astype('uint8'), 180, 180, 30)



