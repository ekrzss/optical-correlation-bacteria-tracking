#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:13:47 2020

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from skimage.feature import peak_local_max

CAMERA_PHOTO = scipy.io.loadmat('camera_photo.mat')
_, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()

INPUT_IMAGE_NUMBER = scipy.io.loadmat('input_image_number.mat')
_, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values()

FILTER_IMAGE_NUMBER = scipy.io.loadmat('filter_image_number.mat')
_, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()

IMAGES = scipy.io.loadmat('inputImages8Bit.mat')
_, _, _, IMAGES = IMAGES.values()
IMAGES = IMAGES[:, :, 0:21]
IMAGES = np.repeat(IMAGES, 21, axis=-1)

# IMAGES_F = scipy.io.loadmat('batchInput.mat')    # The bad ones
# _, _, _, IMAGES_F = IMAGES_F.values()

A, _ = np.where(INPUT_IMAGE_NUMBER > 21)
CAMERA_PHOTO = np.delete(CAMERA_PHOTO, A, axis=-1)
INPUT_IMAGE_NUMBER = np.delete(INPUT_IMAGE_NUMBER, A)
FILTER_IMAGE_NUMBER = np.delete(FILTER_IMAGE_NUMBER , A)

Z = np.argsort(INPUT_IMAGE_NUMBER[0:21])
ZZ = CAMERA_PHOTO[:, :, Z]
ZZZ = INPUT_IMAGE_NUMBER[0:21]
ZZZZ = ZZZ[Z]

CAMERA_PHOTO[:, :, 0:21] = ZZ
INPUT_IMAGE_NUMBER[0:21] = ZZZZ
del Z, ZZ, ZZZ, ZZZZ

# CORR_CPU = np.load('CORR_CPU.npy')
# CAMERA_PHOTO = CORR_CPU;
# CAMERA_PHOTO = CAMERA_PHOTO[380:856, 604:1100, :].astype('float32')

#%
# IMAGE_NUM = 0 # Image number (1, 2,...)
# FILTER_NUM = 0 # Filter number (1, 2,...)          
# SLICE = 21*FILTER_NUM + IMAGE_NUM     # n*22, n=0,1,2... for image/filter pair
# CAM_PHOTO_SLICE = CAMERA_PHOTO[:, :, SLICE] / np.max(CAMERA_PHOTO[:, :, SLICE]) 
# PKS = peak_local_max(A, num_peaks=10, min_distance=10)

# PEAK_NUM=0  # Peak number

def peak_gauss_fit_analysis(normalized_input, peak_number, peak_array, sel_size):
    k = peak_number  # Peak number
    # sel_size = 15
    DATA = normalized_input[peak_array[k][0]-sel_size:peak_array[k][0]+sel_size, peak_array[k][1]-sel_size:peak_array[k][1]+sel_size]
    pks = peak_local_max(DATA, num_peaks=1)
    INTENSITY = DATA[pks[0][0], pks[0][1]]
    
    def gauss(x, x0, y, y0, sigma, MAX):
        # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
        return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))    
    
    I, J = np.meshgrid(np.arange(2*sel_size), np.arange(2*sel_size))
    sig = np.linspace(0.1, 5, 200)
    chisq = np.empty_like(sig)
    
    for ii in range(np.shape(sig)[0]):
            chisq[ii] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], np.max(DATA)))**2) / np.var(DATA)
                
    LOC_MIN = np.where(chisq == np.min(chisq))
    SIGMA_OPT = sig[LOC_MIN[0][0]]
    fitted_gaussian = gauss(I, sel_size, J, sel_size, SIGMA_OPT, np.max(DATA)) #ZZ
    OP = np.sum(DATA)
    
    return INTENSITY, SIGMA_OPT, OP, fitted_gaussian, DATA

# [INTENSITY, SIGMA_OPT, ZZ, DATA] = peak_gauss_fit_analysis(CAM_PHOTO_SLICE, 0, PKS)


#%
SEL_SIZE = 10
NUM_PEAKS = 4
INTENSITY = []
SIGMA = []
OP = []
PEAK_AREA = np.empty((2*SEL_SIZE, 2*SEL_SIZE, NUM_PEAKS, 441))
PEAK_AREA_GAUSS = np.empty((2*SEL_SIZE, 2*SEL_SIZE, NUM_PEAKS, 441))
P = []
PP = []


I = np.empty(NUM_PEAKS)
SIGMA_OPT = np.empty(NUM_PEAKS)  
OPT_POWER = np.empty(NUM_PEAKS)
for jj in range(21):
    
    for ii in range(21):        
        SLICE = 21*jj + ii
        CAM_PHOTO_SLICE = CAMERA_PHOTO[:, :, SLICE] / np.max(CAMERA_PHOTO[:, :, SLICE])
        PKS = peak_local_max(CAM_PHOTO_SLICE, num_peaks=NUM_PEAKS, min_distance=10)
        ZZ= np.empty((2*SEL_SIZE, 2*SEL_SIZE, len(PKS)))
        DATA = np.empty((2*SEL_SIZE, 2*SEL_SIZE, len(PKS)))

        for k in range(len(PKS)):
            print(jj, ii, k)
            I[k], SIGMA_OPT[k], OPT_POWER[k], ZZ[:, :, k], DATA[:, :, k] = peak_gauss_fit_analysis(CAM_PHOTO_SLICE, peak_number=k, peak_array=PKS, sel_size=SEL_SIZE)
            P.append([PEAK_AREA[:, :, k, SLICE]])        
            PP.append([PEAK_AREA_GAUSS[:, :, k, SLICE]])
        
        PEAK_AREA[:, :, :, 21*jj+ii] = DATA
        PEAK_AREA_GAUSS[:, :, :, 21*jj+ii] = ZZ
        INTENSITY = np.concatenate((INTENSITY, I), axis=0)
        SIGMA= np.concatenate((SIGMA, SIGMA_OPT), axis=0)
        OP = np.concatenate((OP, OPT_POWER), axis=0)
        
# P = []
# PP = []
# for k in range(441):
#     for i in range(NUM_PEAKS):
#         P.append([PEAK_AREA[:, :, i, k]])        
#         PP.append([PEAK_AREA_GAUSS[:, :, i, k]])
        
#%
PEAKS = np.tile(np.arange(NUM_PEAKS), 441)    
FI_IM_NUM= np.repeat(np.arange(21), repeats=21*NUM_PEAKS)
IN_IM_NUM = np.repeat(np.tile(np.arange(21), reps=21), NUM_PEAKS)
CAM_PHOTO = np.repeat(np.arange(441), NUM_PEAKS)
    
DF = pd.DataFrame({'CAM_PHOTO':CAM_PHOTO,
                   'INPUT_IMAGE_NUMBER':IN_IM_NUM,
                   'FILTER_NUMBER':FI_IM_NUM,
                   'PEAK_NUMBER': PEAKS,
                   'INTENSITY': INTENSITY,
                   'STD': SIGMA,
                   'OPTICAL_POWER': OP,
                   'PEAK_AREA': P,
                   'PEAK_AREA_GAUSS': PP})

 #%%
       
i=0
DF[i*10:(i+1)*10].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis')
#%%
AA = DF[DF['INPUT_IMAGE_NUMBER'] == DF['FILTER_NUMBER']]
i=0
fig, ax = plt.subplots(3, 2)

AA[i*6:(i+1)*6].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis', ax=ax[0, 0])
AA[(i+1)*6:(i+2)*6].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis', ax=ax[0, 1])
AA[(i+2)*6:(i+3)*6].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis', ax=ax[1, 0])
AA[(i+3)*6:(i+4)*6].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis', ax=ax[1, 1])
AA[(i+4)*6:(i+5)*6].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis', ax=ax[2, 0])
AA[(i+5)*6:(i+6)*6].plot.scatter(x='INTENSITY', y='OPTICAL_POWER', c='STD', colormap='viridis', ax=ax[2, 1])

#%%
IM_NUM = 0
PK_NUM = 0
fig, ax = plt.subplots(4, 2, constrained_layout=True)

ax[0, 0].imshow(DF.PEAK_AREA[0][0])
ax[0, 0].set_title('Image number '+np.str(IM_NUM)+' with Peak number '+np.str(PK_NUM))

ax[0, 1].imshow(DF.PEAK_AREA_GAUSS[0][0])
plt.title('STD: '+np.str(IM_NUM)+' with Intensity '+np.str(PK_NUM))

ax[1, 0].imshow(DF.PEAK_AREA[1][0])
plt.title('Image number '+np.str(IM_NUM)+' with Peak number '+np.str(PK_NUM))

ax[1, 1].imshow(DF.PEAK_AREA_GAUSS[1][0])
plt.title('STD: '+np.str(IM_NUM)+' with Intensity '+np.str(PK_NUM))

ax[2, 0].imshow(DF.PEAK_AREA[2][0])
plt.title('Image number '+np.str(IM_NUM)+' with Peak number '+np.str(PK_NUM))

ax[2, 1].imshow(DF.PEAK_AREA_GAUSS[2][0])
plt.title('STD: '+np.str(IM_NUM)+' with Intensity '+np.str(PK_NUM))

ax[3, 0].imshow(DF.PEAK_AREA[3][0])
plt.title('Image number '+np.str(IM_NUM)+' with Peak number '+np.str(PK_NUM))

ax[3, 1].imshow(DF.PEAK_AREA_GAUSS[3][0])
plt.title('STD: '+np.str(IM_NUM)+' with Intensity '+np.str(PK_NUM))

plt.show()


#%%
# PEAK_NUM = 0
# plt.suptitle('Image number '+np.str(IMAGE_NUM)+' with Filter number '+np.str(FILTER_NUM))
# plt.subplot(2, 2, 1)
# plt.title('Original Image')
# plt.imshow(IMAGES[:, :, SLICE], cmap='gray')

# plt.subplot(2, 2, 2)
# plt.title('Correlation with peaks marked')
# plt.imshow(CAM_PHOTO_SLICE)

# for ii, txt in enumerate(np.arange(np.shape(PKS)[0])):
#     plt.annotate(txt, (PKS[ii, 1], PKS[ii, 0]))
# plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', color='r', facecolors='none')

# plt.subplot(2,2,3)
# plt.title('Fitted Gaussian for peak '+np.str(PEAK_NUM)+' and ' +r'$\sigma=$'+np.str(SIGMA_OPT[PEAK_NUM].astype('float16')))
# plt.imshow(ZZ[:, :, PEAK_NUM])    

# plt.subplot(2,2,4)
# plt.title('Raw data for peak '+np.str(PEAK_NUM))
# plt.imshow(DATA[:, :, PEAK_NUM])    
        
# plt.show()