#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:02:39 2020

@author: erick
"""
import numpy as np
import matplotlib as mpl
mpl.rc('figure',  figsize=(10, 6))
import matplotlib.pyplot as plt
import scipy.io
import easygui as gui

#%% Select camera_photo, input_image_number and filter_image_number arrays
PATHS = gui.fileopenbox(msg='Select File', title='Files', default='/media/erick/NuevoVol/LINUX LAP/PhD', filetypes='.mat', multiple='True')

#%% For Ground truth
CAMERA_PHOTO = scipy.io.loadmat(PATHS[0])
_, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()

INPUT_IMAGE_NUMBER = scipy.io.loadmat(PATHS[2])
_, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values() 
INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER[0, :]

FILTER_IMAGE_NUMBER = scipy.io.loadmat(PATHS[1])
_, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()
FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER[0, :]

Z = np.argsort(INPUT_IMAGE_NUMBER[0:21])
ZZ = CAMERA_PHOTO[:, :, Z]
ZZZ = INPUT_IMAGE_NUMBER[0:21]
ZZZZ = ZZZ[Z]

CORR_CPU = np.load('CORR_CPU.npy')
CAMERA_PHOTO = CAMERA_PHOTO[330:1076, 332:1037, :].astype('float32')
#%%
# CAMERA_PHOTO = scipy.io.loadmat('camera_photo.mat')
# _, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()

# INPUT_IMAGE_NUMBER = scipy.io.loadmat('input_image_number.mat')
# _, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values()

# FILTER_IMAGE_NUMBER = scipy.io.loadmat('filter_image_number.mat')
# _, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()

# A, _ = np.where(INPUT_IMAGE_NUMBER > 21)
# CAMERA_PHOTO = np.delete(CAMERA_PHOTO, A, axis=-1)
# INPUT_IMAGE_NUMBER = np.delete(INPUT_IMAGE_NUMBER, A)
# FILTER_IMAGE_NUMBER = np.delete(FILTER_IMAGE_NUMBER , A)

# Z = np.argsort(INPUT_IMAGE_NUMBER[0:21])
# ZZ = CAMERA_PHOTO[:, :, Z]
# ZZZ = INPUT_IMAGE_NUMBER[0:21]
# ZZZZ = ZZZ[Z]

# CAMERA_PHOTO[:, :, 0:21] = ZZ
# INPUT_IMAGE_NUMBER[0:21] = ZZZZ
# del Z, ZZ, ZZZ, ZZZZ

# CORR_CPU = np.load('CORR_CPU.npy')
# CAMERA_PHOTO = CAMERA_PHOTO[380:856, 604:1100, :].astype('float32')
#%%
# Check system "ideal" filter-image combination
# Not all combination have good correlation image
for i in range(21):
#i = 20
    k = 22*i
    plt.figure(i+1)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(CORR_CPU[:, :, k] * 10**-7)
    L = np.where(CORR_CPU[:, :, k] == np.max(CORR_CPU[:, :, k]))
    LOCS0, LOCS1 = L[0][0], L[1][0]
    ax[0].scatter(LOCS1, LOCS0, marker='o', color='r', facecolors='none')
    ax[0].set_title('CPU: '+np.str(i+1))
    
    ax[1].imshow(CAMERA_PHOTO[:, :, k])
    L = np.where(CAMERA_PHOTO[:, :, k] == np.max(CAMERA_PHOTO[:, :, k]))
    LOCS00, LOCS11 = L[0][0], L[1][0]
    ax[1].scatter(LOCS11, LOCS00, marker='o', color='r', facecolors='none')
    ax[1].set_title('Optalysys: '+np.str(i+1))
    
    plt.show()