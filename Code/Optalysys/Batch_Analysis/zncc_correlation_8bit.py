# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 14:51:34 2021

@author: eers500
"""

import glob
import numpy as np
import matplotlib as mpl
mpl.rc('figure',  figsize=(10, 6))
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f
import time
import easygui as gui
from scipy import ndimage
from numba import vectorize, jit
from tqdm import tqdm

#%% Import Video correlate
path_vid = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
VID = f.videoImport(path_vid, 0)
ni, nj, nk = np.shape(VID)
# MAX_VID = np.max(VID)
# VID = np.uint8(255*(VID / MAX_VID))

#%% Import LUT form images
path_lut = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
LUT = f.videoImport(path_lut, 0)
mi, mj, mk = np.shape(LUT)

#%% Prepare arrays
VID_zn = np.empty_like(VID)
for k in range(nk):
    A = VID[:,:,k]
    VID_zn[:, :, k] = (A-np.mean(A))/np.std(A)

LUT_zn = np.empty_like(LUT)
for k in range(mk):
    A = LUT[:,:,k]
    LUT_zn[:, :, k] = (A-np.mean(A))/np.std(A)

# VID_zn = VID
# LUT_zn = LUT
# A = np.repeat(VID_zn, repeats=mk, axis=-1)
# B = np.tile(LUT_zn, nk)

#%% Correltion in GPU
#@vectorize(["complex128(complex128, complex128)"], target='cuda')   #not good
@jit(nopython=True) 
def corr_gpu(a, b):
    return a*np.conj(b)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

# BB = np.empty_like(A)
# for k in range(np.shape(B)[2]):
#     print(k)
#     # BB[:, :, k] = np.pad(B[:, :, k], int((1024-110)/2))
#     BB[:, :, k] = np.pad(B[:, :, k], int((A.shape[0]-B.shape[0])/2))
# del B
FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

# C = np.empty_like(A, dtype='float32')
# T0 = time.time()
# T = []
# for k in tqdm(range(nk*mk)):
# # for k in range(100):
#     # print(k)
#     # BFT = FT(BB[:,:,k])
#     BFT = FT(np.pad(B[:, :, k], int((A.shape[0]-B.shape[0])/2)))
#     R = corr_gpu(FT(A[:, :, k]), BFT).astype('complex64')
#     # C[:, :, k] = np.abs(IFT(R))
#     C[:, :, k] = np.abs(IFT(R / np.sum(BFT)))    # Normalize with sum of pixel values of filters
#     T.append((time.time()-T0)/60)
# print(T[-1])

C = np.empty((ni, nj, nk*mk))
T0 = time.time()
T = []
for i in tqdm(range(nk)):
    im = VID_zn[:, :, i]
    imft = FT(im)
    for j in range(mk):
        fm = np.pad(LUT_zn[:, :, j], int((ni-mi)/2))
        fmft = FT(fm)
        C[:, :, i*mk+j] = np.abs(IFT(corr_gpu(imft, fmft)/np.sum(fmft)))
        T.append((time.time()-T0)/60)
print(T[-1])
       
#%% Divide by median image
# CC = np.empty_like(C)
# for i in tqdm(range(nk)):
#     med = f.medianImage(C[:, :, i*mk:i*mk+mk], mk)
#     med = np.expand_dims(med, 2)
#     med = np.repeat(med, mk, 2)
#     CC[:, :, i*mk:i*mk+mk] = C[:, :, i*mk:i*mk+mk] / med 

#%%
# CC = C
# n = 0
# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
# ax[0].imshow(C[:,:,n])
# ax[1].imshow(CC[:,:,n])

# f.surf(C[:, :, n])
# f.surf(CC[:, :, n])


#%% Calculate correlation in CPU
# from skimage.feature import match_template

# C = np.empty((ni, nj, nk*mk))
# T0 = time.time()
# T = []
# for i in tqdm(range(15)):
#     # im = VID_zn[:, :, i]
#     # imft = FT(im)
#     for j in range(mk):
#         fm = np.pad(LUT_zn[:, :, j], int((ni-mi)/2))
#         # fmft = FT(fm)
#         # CORR[:, :, i*mk+j] = np.abs(IFT(corr_gpu(imft, fmft)/np.sum(fmft)))
#         C[:, :, i*mk+j] = match_template(VID_zn[:, :, i], LUT_zn[:,:,j], pad_input=True)
#         T.append((time.time()-T0)/60)
# print(T[-1])
