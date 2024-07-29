#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:59:50 2019

@author: erick
"""

import numpy as np
import matplotlib as mpl
mpl.rc('figure',  figsize=(10, 6))
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import functions as f
import easygui as gui
from skimage.feature import peak_local_max

#%
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/'
PATHS = gui.fileopenbox(msg='Select File',
                        title='Files',
                        # default='/home/erick/Documents/PhD/Correaltion_Project/Optalysys/Batch_Analysis/',
                        # default='/media/erick/NuevoVol/LINUX_LAP/PhD/Optical_Correlation_Results/',
                        default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/',
                        filetypes='.mat', 
                        multiple='True')


number_of_images, number_of_filters = gui.multenterbox(msg='How much images and filters?',
                            title='Number of images and filters',
                            fields=['Number of images:',
                                   'Number of filters:']) 

number_of_images = int(number_of_images)
number_of_filters = int(number_of_filters)

#%% Read MAT files
CAMERA_PHOTO = scipy.io.loadmat(PATHS[0])
_, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()

INPUT_IMAGE_NUMBER = scipy.io.loadmat(PATHS[2])
_, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values() 
INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER[0, :]

FILTER_IMAGE_NUMBER = scipy.io.loadmat(PATHS[1])
_, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()
FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER[0, :]

#%%
# import mat73
# data_dict = mat73.loadmat(PATHS[0])

#%%
import h5py
arrays = {}
f = h5py.File(PATHS[0])
for k, v in f.items():
    arrays[k] = np.array(v)
    
CAMERA_PHOTO = arrays['camera_photo']
CAMERA_PHOTO = np.swapaxes(CAMERA_PHOTO, 0, 1)
CAMERA_PHOTO = np.swapaxes(CAMERA_PHOTO, 1, 2)

#%% For Colloids
# Z = np.argsort(INPUT_IMAGE_NUMBER[0:21])
# ZZ = CAMERA_PHOTO[:, :, Z]
# ZZZ = INPUT_IMAGE_NUMBER[0:21]
# ZZZZ = ZZZ[Z]

# CAMERA_PHOTO[:, :, 0:21] = ZZ
# INPUT_IMAGE_NUMBER[0:21] = ZZZZ
# del Z, ZZ, ZZZ, ZZZZ
#%% Order arrays according to image-filter combination
ni, nj, nk = np.shape(CAMERA_PHOTO)
cam = np.empty_like(CAMERA_PHOTO)
# number = int(np.sqrt(np.shape(CAMERA_PHOTO)[2]))
number = number_of_images*number_of_filters
for k in range(number):
    i_filt = np.where(FILTER_IMAGE_NUMBER==k+1)[0]
    # images_number = INPUT_IMAGE_NUMBER[i_filt]
    filter_temp = FILTER_IMAGE_NUMBER[i_filt]
    images_temp = INPUT_IMAGE_NUMBER[i_filt]
    i_images = np.argsort(images_temp)
    frames = CAMERA_PHOTO[:, :, i_filt]
    # cam[:, :, number*k:number*k+number] = frames[:, :, i_images]
    cam[:, :, number_of_images*k:number_of_images*k+number_of_images] = frames[:, :, i_images]

CAMERA_PHOTO = cam
del cam
#%% Check for max correlation of first input image
# CAM = CAMERA_PHOTO[:, :, :21]
# MAX = np.max(CAM, axis=(0, 1))
# MAX_ID = np.where(MAX == MAX.max())[0][0]

#%%
# CAMERA_PHOTOs = scipy.io.loadmat('camera_photo.mat')
# _, _, _, CAMERA_PHOTOs = CAMERA_PHOTOs.values()

# FILTERS = scipy.io.loadmat('filters.mat')
# _, _, _, FILTERS = FILTERS.values()

# INPUT_IMAGE_NUMBER = scipy.io.loadmat('input_image_number.mat')
# _, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values()

# FILTER_IMAGE_NUMBER = scipy.io.loadmat('filter_image_number.mat')
# _, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()

# A, _ = np.where(INPUT_IMAGE_NUMBER > 21)
# CAMERA_PHOTO = np.delete(CAMERA_PHOTOs, A, axis=-1)
# INPUT_IMAGE_NUMBER = np.delete(INPUT_IMAGE_NUMBER, A)
# FILTER_IMAGE_NUMBER = np.delete(FILTER_IMAGE_NUMBER , A)

# INPUT_IMAGE = scipy.io.loadmat('inputImagesBinary.mat')
# _, _, _, INPUT_IMAGE = INPUT_IMAGE.values()
 
# TARGET_IMAGE = scipy.io.loadmat('target_images_binary.mat')
# _, _, _, TARGET_IMAGE = TARGET_IMAGE.values()

# BATCH_INPUT = scipy.io.loadmat('batchInput.mat')    # The ones that project all slices
# _, _, _, BATCH_INPUT = BATCH_INPUT.values()

# Z = np.argsort(INPUT_IMAGE_NUMBER[0:21])
# ZZ = CAMERA_PHOTO[:, :, Z]
# ZZZ = INPUT_IMAGE_NUMBER[0:21]
# ZZZZ = ZZZ[Z]

# CAMERA_PHOTO[:, :, 0:21] = ZZ
# INPUT_IMAGE_NUMBER[0:21] = ZZZZ
# del Z, ZZ, ZZZ, ZZZZ

#%%
# k=20
# plt.subplot(1, 2, 1)
# plt.imshow(INPUT_IMAGE[:, :, k], cmap='gray')

# plt.subplot(1, 2, 2)
# plt.imshow(IN[:, :, k], cmap='gray')

# plt.show()
#%% Crop array
#CAMERA_PHOTO = CAMERA_PHOTO[246:948, 482:1163, :]
#CAMERA_PHOTO = CAMERA_PHOTO[272:272+510, 482:482+512, :]
# s = np.sum(CAMERA_PHOTO**2, axis=(0, 1))
# fi = np.sum(np.repeat(FILTERS**2, 21, axis=-1), axis=(0, 1))
# CAMERA_PHOTO = CAMERA_PHOTO[380:856, 604:1100, :].astype('float32')

# for k in range(441):
#     CAMERA_PHOTO[:, :, k] = CAMERA_PHOTO[:, :, k] / (s[k] + fi[k])

# Substract background
#MEAN = np.median(CAMERA_PHOTO, axis=-1)
#for k in range(441):
#    CAMERA_PHOTO[:, :, k] = CAMERA_PHOTO[:, :, k] / MEAN


#%%
# INDEX = 6
# PKS = peak_local_max(CAMERA_PHOTO[:, :, INDEX], min_distance=1, threshold_abs=15)
#
# plt.imshow(CAMERA_PHOTO[:, :, INDEX])
# plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', facecolors='none', s=80, edgecolors='r')
# plt.show()

#%% Pixel normalization
# for j in range(np.shape(CAMERA_PHOTO)[2]):
# #    CAMERA_PHOTO[:, :, j] = CAMERA_PHOTO[:, :, j] / (np.sum(np.real(FILTERS[:, :, FILTER_IMAGE_NUMBER[j]-1]))*np.sum(np.abs(CAMERA_PHOTO[:, :, j])))
#     CAMERA_PHOTO[:, :, j] = CAMERA_PHOTO[:, :, j] / np.sum(CAMERA_PHOTO[:, :, j])**2

#%% Normalization to compare images
# SUM_FILTS = np.empty(np.shape(CAMERA_PHOTO)[2])
# SUM_CAM = np.empty_like(SUM_FILTS)
# MULT = np.empty_like(SUM_FILTS)
# CAMERA_PHOTOS = np.empty_like(CAMERA_PHOTO)

# # FILTERS[FILTERS <= 0] = 0
# # FILTERS[FILTERS > 0] = 255


# # for i in range(np.shape(CAMERA_PHOTO)[2]):
# #     # SUM_FILTS[i] = np.sum(np.real(FILTERS[:, :, FILTER_IMAGE_NUMBER[i]-1]))
# #     SUM_CAM[i] = np.sum(np.real(CAMERA_PHOTO[:, :, i]))
# #     # MULT[i] = SUM_FILTS[i] + SUM_CAM[i]
# #     # CAMERA_PHOTOS[:, :, i] = CAMERA_PHOTO[:, :, i] / MULT[i]
# #     CAMERA_PHOTOS[:, :, i] = CAMERA_PHOTO[:, :, i]

# SUM_CAM = np.sum(CAMERA_PHOTO, axis=(0, 1))
# SUM_FILT = np.sum(FILTERS, axis=(0, 1))
# CAMERA_PHOTOS = CAMERA_PHOTO / (np.max(SUM_CAM) + np.max(SUM_FILT))
# CAMERA_PHOTO = CAMERA_PHOTOS

# # plt.figure()
# # plt.plot(SUM_FILTS)
# #
# # plt.figure()
# # plt.plot(SUM_CAM)
# #
# # plt.figure()
# # plt.plot(MULT)
# #
# # plt.figure()
# # plt.imshow(CAMERA_PHOTOS[:, :, 0])


#%% Histogram equalization and normalization
# CAMS, cdf = f.histeq(CAMERA_PHOTO)

#%% Get (x,y) coordinates of maximum correlation spots
#CORR = np.empty((np.shape(CAMERA_PHOTO)[0], np.shape(CAMERA_PHOTO)[1] , 441), dtype='float32')

# To run from numpy array file of correlation results
# CAMERA_PHOTO = np.load('CORR_GPU_Colloids_P1.npy')   # Same directory as script

# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/'
# CAMERA_PHOTO = np.load(path+'substack_GPU_Result.npy')
# CAMERA_PHOTO = np.load(path+'substack_CES_GPU_Result.npy')
# CAMERA_PHOTO = np.load(path+'substack_CPU_Result.npy')
# CAMERA_PHOTO = np.load(path+'substack_CES_CPU_Result.npy')

#%%
# number_of_images = number
# number_of_filters = number

# number_of_images = 31
# number_of_filters = 21

#%% Transpose CAMERA_PHOTO if flipped
s = np.shape(CAMERA_PHOTO)
cam = np.zeros((s[1], s[0], s[2]), dtype='uint8')
for k in range(np.shape(CAMERA_PHOTO)[-1]):
    cam[:, :, k] = np.transpose(CAMERA_PHOTO[:, :, k])
    
CAMERA_PHOTO = cam
del cam

#%%
MAX = []
LOCS = np.empty((np.shape(CAMERA_PHOTO)[2], 2))
for k in range(np.shape(CAMERA_PHOTO)[2]):
    MAX.append(np.max(CAMERA_PHOTO[:, :, k]))
    L = np.where(CAMERA_PHOTO[:, :, k] == np.max(CAMERA_PHOTO[:, :, k]))
    LOCS[k, 0], LOCS[k, 1] = L[0][0], L[1][0]

#  Columns are the input images and rows are the input filters
MAX = np.reshape(MAX, (number_of_filters, number_of_images), 'F')
Li, Lj = LOCS[:, 0], LOCS[:, 1]

Li = np.reshape(Li, (number_of_filters, number_of_images), 'F')
Lj = np.reshape(Lj, (number_of_filters, number_of_images), 'F')

#%
# Get maximum correlation filter for all images    
MAX_FILT = np.empty(number_of_images) 
pos_i = []
pos_j = []
for i in range(number_of_images):
    # M = MAX[i*21:i*21+21]
    # M = np.array(M)
    # MAX_FILT[i] = np.where(np.max(M) == M)[0][0]
    MAX_FILT[i] = np.where(MAX[:, i] == MAX[:, i].max())[0][0]
    pos_i.append(Li[int(MAX_FILT[i]), i])
    pos_j.append(Lj[int(MAX_FILT[i]), i])
    
    

plt.figure()
plt.imshow(MAX, cmap='viridis')
plt.xlabel('Image Number', fontsize=15)
plt.ylabel('Filter Number', fontsize=15)
plt.title('Image Filter Combination Max Correlation P4', fontsize=20)
plt.colorbar()
plt.show()
print(MAX_FILT)

COMB = np.transpose(np.vstack((INPUT_IMAGE_NUMBER, FILTER_IMAGE_NUMBER)))
# f.imshow_sequence(CAMERA_PHOTO, 0.1, 1)

#%% Fit local values resolution improvement
# from scipy.optimize import curve_fit

# # def func(x, a, b, c):
# #     return a*x**2 + b*x + c

# def func(x, MAX, x0, sigma):
#     return MAX * np.exp(-(x-x0)**2 / (2*sigma**2))

# plt.figure()
# MMAX = np.pad(MAX, ((3, 3), (0, 0)), 'linear_ramp')
# for k in range(len(MAX)):
#     ydata = MMAX[:, k]
#     xdata = np.arange(len(ydata))
    
#     plt.cla()
#     plt.plot(xdata, ydata, 'bo-', label='data')
    
#     popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [200, 10, 1.2]))
#     xfit = np.linspace(0, len(ydata), 100)
#     yfit = func(xfit, *popt)
#     plt.plot(xfit, yfit, 'g-', label='fit')
    
#     max_bool = yfit == yfit.max()
#     plt.plot(xfit[max_bool], yfit[max_bool] , 'r*', 
#               label='new max at = (%5.3f, %5.3f)' % tuple([xfit[max_bool], yfit[max_bool]]))
#     plt.xlabel('Filter', fontsize=15)
#     plt.ylabel('Pixel Value', fontsize=15)
    
#     plt.legend()
#     plt.show()
#     plt.pause(1)
#     # plt.savefig(np.str(k)+'.png')
#     print(popt)

#%% Optimize measurement using local values fit
from scipy.optimize import curve_fit

# def func(x, a, b, c):
#     return -a*x**2 + b*x + c

def func(x, MAX, x0, sigma):
    return MAX * np.exp(-(x-x0)**2 / (2*sigma**2))

padval = 3    # value >1
table = []
plt.figure()
MMAX = np.pad(MAX, ((padval, padval), (0, 0)), 'wrap')
for k in range(MMAX.shape[1]):
    # plt.cla()
    ydata = MMAX[:, k]
    xdata = np.arange(len(ydata))
    yy = ydata[k:k+2*padval+1]
    xx = np.linspace(-int(np.sqrt(padval))+k, int(np.sqrt(padval))+k, len(yy))
    popt, pcov = curve_fit(func, xx, yy, bounds=(0, [200, 20, 10]))
    xxfit = np.linspace(-2+k, 2+k, 100)
    yyfit = func(xxfit, *popt)
    max_max = yyfit == yyfit.max()
    plt.plot(xxfit[max_max][0], yyfit[max_max][0] , 'rH', 
                  label='new max at = (%5.3f, %5.3f)' % tuple([xxfit[max_max][0], yyfit[max_max][0]]))
    plt.vlines(xxfit[max_max][0], yyfit[max_max][0], 255, 'red')
    plt.plot(xx, yy, 'g.-')
    plt.plot(xxfit, yyfit, 'b-')
    plt.grid()
    plt.show()
    plt.pause(0.5)
    table.append(popt)
    print(popt)
plt.legend()

#%% Fit for resolution improvement using all data
# from scipy.optimize import curve_fit

# # def func(x, a, b, c):
# #     return a*x**2 + b*x + c

# def func(x, MAX, x0, sigma):
#     return MAX * np.exp(-(x-x0)**2 / (2*sigma**2))

# plt.figure()
# # MMAX = np.pad(MAX, ((3, 3), (0, 0)), 'reflect')
# for k in range(len(MAX)):
#     ydata = MAX[:, k]
#     xdata = np.arange(len(ydata))
    
#     if k>1:
#         plt.cla()
#         plt.plot(xdata, ydata, 'bo-', label='data')
        
#         popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [200, 10, 1.2]))
#         xfit = np.linspace(0, len(ydata)-1, 100)
#         yfit = func(xfit, *popt)
#         plt.plot(xfit, yfit, 'g-', label='fit')
        
#         max_bool = yfit == yfit.max()
#         plt.plot(xfit[max_bool], yfit[max_bool] , 'r*', 
#                   label='new max at = (%5.3f, %5.3f)' % tuple([xfit[max_bool], yfit[max_bool]]))
#         plt.xlabel('Filter', fontsize=15)
#         plt.ylabel('Pixel Value', fontsize=15)
        
#         plt.legend()
#         plt.show()
#         plt.pause(1)
#         # plt.savefig(np.str(k)+'.png')
#         print(popt)
#%%
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Make data
# ra = np.arange(1024)
# x, y = np.meshgrid(622, 708)

# # Plot the surface
# ax.plot_surface(x, y, CAMERA_PHOTO[:,:,0], rstride=1, cstride=1, cmap=cm.viridis_r)

# plt.show()

#%% Export results data to CSV
# results = pd.DataFrame()
# results['P4-5'] = MAX_FILT

# file_name = '/home/erick/Documents/PhD/Correaltion_Project/Optalysys/Batch_Analysis/June_04_2020_GT_data/GT_results.txt'
# file_name = '/media/erick/NuevoVol/LINUX_LAP/PhD/Optical_Correlation_Results/Colloids_results.csv'
# results.to_csv(file_name, index=False)

#%% Plot images with coordinates of maximum correlation    
# k = 20   
# plt.imshow(CAMERA_PHOTO[:, :, k*22], cmap='gray')
# plt.scatter(LOCS[k*21+k,1], LOCS[k*21+k, 0], marker='o', color='r', facecolors='none')
# plt.show()

#%% Plot images with maximum values signaled
# k = 0         
# plt.imshow(CAMERA_PHOTO[:, :, k], cmap='jet')
# plt.scatter(LOCS[k,1], LOCS[k, 0], marker='o', color='r', facecolors='none')
# plt.show()
#%% Plotly surface plot
# import plotly.graph_objects as go
# from plotly.offline import plot

# fig = go.Figure(data=[go.Surface(z=MAX)])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                   highlightcolor="limegreen", project_z=True))
# fig.update_layout(title='correlation')
# fig.show()
# plot(fig)

