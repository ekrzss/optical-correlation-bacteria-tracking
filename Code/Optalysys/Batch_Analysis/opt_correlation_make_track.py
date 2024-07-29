# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:32:39 2021

@author: eers500
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
import time
from tqdm import tqdm

#%%
mode = True   # Set False to select GPU .npy arrays

if mode: 
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
    
    #% Read MAT files
    CAMERA_PHOTO = scipy.io.loadmat(PATHS[0])
    _, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()
    
    INPUT_IMAGE_NUMBER = scipy.io.loadmat(PATHS[2])
    _, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values() 
    INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER[0, :]
    
    FILTER_IMAGE_NUMBER = scipy.io.loadmat(PATHS[1])
    _, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()
    FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER[0, :]
    
    COMB = np.transpose(np.vstack((INPUT_IMAGE_NUMBER, FILTER_IMAGE_NUMBER)))
    

else:
    
    PATHS = gui.fileopenbox(msg='Select File',
                            title='Files',
                            # default='/home/erick/Documents/PhD/Correaltion_Project/Optalysys/Batch_Analysis/',
                            # default='/media/erick/NuevoVol/LINUX_LAP/PhD/Optical_Correlation_Results/',
                            default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/',
                            filetypes='.npz', 
                            multiple='True')
    
    number_of_images, number_of_filters = gui.multenterbox(msg='How much images and filters?',
                                title='Number of images and filters',
                                fields=['Number of images:',
                                       'Number of filters:'])
    
    number_of_images = int(number_of_images)
    number_of_filters = int(number_of_filters)
    
    CAMERA_PHOTO = np.load(PATHS[0])
    CAMERA_PHOTO = CAMERA_PHOTO['a']
    
    a = np.arange(1, number_of_images+1)
    b = np.arange(1, number_of_filters+1)
    
    INPUT_IMAGE_NUMBER = np.repeat(a, repeats=number_of_filters)
    FILTER_IMAGE_NUMBER = np.tile(b, reps=number_of_images)
    
#%%
# import h5py
# arrays = {}
# f = h5py.File(PATHS[0])
# for k, v in f.items():
#     arrays[k] = np.array(v)
    
# CAMERA_PHOTO = arrays['camera_photo']
# CAMERA_PHOTO = np.swapaxes(CAMERA_PHOTO, 0, 1)
# CAMERA_PHOTO = np.swapaxes(CAMERA_PHOTO, 1, 2)

#%% Transpose CAMERA_PHOTO if flipped
# s = np.shape(CAMERA_PHOTO)
# cam = np.zeros((s[1], s[0], s[2]), dtype='uint8')
# for k in range(np.shape(CAMERA_PHOTO)[-1]):
#     cam[:, :, k] = np.transpose(CAMERA_PHOTO[:, :, k])
    
# CAMERA_PHOTO = cam
# del cam

#%% Order array according to image number combination with all filters
if mode:
    cam = np.empty_like(CAMERA_PHOTO)
    filter_num = np.empty_like(FILTER_IMAGE_NUMBER)   
    input_num = np.empty_like(INPUT_IMAGE_NUMBER)  
    
    for k in range(number_of_images):
        index_image = INPUT_IMAGE_NUMBER == k+1
        filter_num[k*number_of_filters:k*number_of_filters+number_of_filters] = FILTER_IMAGE_NUMBER[index_image]
        input_num[k*number_of_filters:k*number_of_filters+number_of_filters] = INPUT_IMAGE_NUMBER[index_image]
        cam[:, :, k*number_of_filters:k*number_of_filters+number_of_filters] = CAMERA_PHOTO[:, :, index_image]
    
    CAMERA_PHOTO = cam
    del cam
    
    comb = np.transpose(np.vstack((input_num, filter_num)))
    
#%% Export CAMERA_PHOTO as 2D txt
# ni, nj, nk = CAMERA_PHOTO.shape

# CC = np.empty((1, nj), dtype='uint8')
# for k in range(nk):
#     CC = np.vstack((CC, CAMERA_PHOTO[:, :, k]))
#     print(k)
# CC = CC[1:, :]    
# np.savetxt('F:\PhD\Archea_LW\Results_CES\correlation_2D.txt', CC, fmt='%i', delimiter=',')

#%%
from skimage.feature import peak_local_max

frame = 0
pk = peak_local_max(CAMERA_PHOTO[:,:,frame], min_distance=25, threshold_rel=0.6)
# pk = peak_local_max(CAMERA_PHOTO[:, :, frame], min_distance=25, threshold_abs=50)
plt.imshow(CAMERA_PHOTO[:,:,frame])
plt.scatter(pk[:, 1], pk[:, 0], c='r', marker='o', alpha=0.6)

#%%
r = [[72, 114], [169, 30], [200, 32], [107, 118]]
r0 = r[0]
epsilon = 5
# peaks = np.empty(number_of_images*number_of_filters, dtype='object')
peak_data = []
ds = 15

# for i in tqdm(range(10)):
for i in tqdm(range(number_of_images)):
    for j in range(number_of_filters):
        temp = CC[:, :, i*number_of_filters+j]
        pks = peak_local_max(temp, min_distance=20, threshold_abs=0.3)
        dist = np.sqrt(np.sum((pks-r0)**2, axis=1))
        dist_bool = dist <= epsilon
        pks = pks[dist_bool]
        
        if len(pks) > 0:
            std = []
            for k in range(len(pks)):
                t = temp[pks[k][0]-ds:pks[k][0]+ds, pks[k][1]-ds:pks[k][1]+ds]
                fit = f.peak_gauss_fit_analysis(t)
                std = fit[1]
                peak_data.append([pks[k][0], pks[k][1], temp[pks[k][0], pks[k][1]], i, j, std])
                r0 = [pks[k][0], pks[k][1]]
                

peak_data = np.array(peak_data)

#% track detection
idmin = []
idmax = []
images = np.unique(peak_data[:, 3])
for image_num in tqdm(images):
    temp = peak_data[peak_data[:, 3] == image_num]
    # idmin.append(np.where(temp[:, 5] == temp[:, 5].min())[0][0])
    # idmax.append(np.where(temp[:, 2] == temp[:, 2].max())[0][0])
    imin = np.where(temp[:, 5] == temp[:, 5].min())[0][0]
    imax = np.where(temp[:, 2] == temp[:, 2].max())[0][0]
    
    idmin.append(temp[imin, :])
    idmax.append(temp[imax, :])
        
idmin = np.array(idmin)
idmax = np.array(idmax)
#%
# plt.subplot(1,2,1)
# plt.plot(idmin[:, 1], -idmin[:, 0], '.-')
# plt.subplot(1,2,2)
# plt.plot(idmax[:, 1], -idmax[:, 0], '.-')
# plt.show()

    
#% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
# ax1.scatter(peaks[:, 0], peaks[:, 1], peaks[:, 4], '.', s=1)
ax1.scatter(idmin[:, 0], idmin[:, 1], idmin[:, 4], '.', c=idmin[:, 3])
pyplot.show() 
    
#%% For Archea to create tracks
# Find peaks for each combination of filter and images and store them in an array pks
# index i correspond to images
# index j correspond to filters
# for each (i, j) in pks there is a collection of peaks for each combination of image-filter
# from intensipy import Intensify
# model = Intensify()

# def normalize_slices(array_3d):
#     nk = array_3d.shape[-1]
#     norm_array = np.empty_like(array_3d)
#     for k in range(nk):
#         norm_array[:, :, k] = array_3d[:, :, k] / array_3d[:, :, k].max()
        
#     return norm_array
    
# def normalize_total(array_3d):
#     nk = array_3d.shape[-1]
#     norm_array = np.empty_like(array_3d)
#     for k in range(nk):
#         norm_array[:, :, k] = array_3d[:, :, k] / array_3d[:, :, k].sum()
        
#     return norm_array

# number_of_images = 40
pks = np.empty((number_of_images, number_of_filters), dtype=object)
pksi = np.zeros((number_of_images, number_of_filters), dtype='int')
pksj = np.zeros((number_of_images, number_of_filters), dtype='int')
pks_vals = -10*np.ones((number_of_images, number_of_filters))

std_dev = np.nan*np.ones((number_of_images, number_of_filters))
max_val = np.nan*np.ones_like(std_dev)
fit = np.empty((number_of_images, number_of_filters), dtype='object')
data = np.empty((number_of_images, number_of_filters), dtype='object')
# Archea
# r0 = [327, 471]
# r0 = [356, 536]
# r0 = [86, 125]
r0 = [70, 113]   # For npz array of 44x44 filter
# r0 = [187, 35]     # For particle 2

# # Ecoli
# r0 = [220, 263]
# ri = np.zeros((number_of_images, number_of_filters), dtype='int')
# rj = np.zeros((number_of_images, number_of_filters), dtype='int')

epsilon = 5
T0 = time.time()
T = np.empty(number_of_images)

for i in tqdm(range(number_of_images)):
# for i in range(10):
    temp_cam = CAMERA_PHOTO[:, :, i*number_of_filters:i*number_of_filters+number_of_filters]
    # temp_cam = normalize_slices(temp_cam)
    # temp_cam = normalize_total(temp_cam)
    # temp_cam = temp_cam / temp_cam.max()
    # temp_cam, cdf = f.histeq(temp_cam)

    # Histogram equalization of every slice    
    # tt = np.empty_like(temp_cam)
    # for jj in range(temp_cam.shape[-1]):
    #     tt[:, :, jj], _ = f.histeq(temp_cam[:, :, jj])
    # temp_cam = tt

    temp_pks = np.zeros((1, 3))

#%
    for j in range(number_of_filters):
# %
        # j=1
        # temp_pks= peak_local_max(temp_cam[:, :, j], num_peaks=1)
        temp_pks = peak_local_max(temp_cam[:,:,j], min_distance=25, threshold_rel=0.5)
        # temp_pks = peak_local_max(temp_cam[:,:,j], min_distance=25, threshold_abs=0.5)
        # tt = np.where(temp_cam[:, :, j]==temp_cam[:, :, j].max())
        # temp_pks = np.array([[tt[0][0], tt[1][0]]])
        # print(temp_pks)
        
        if temp_pks.shape[0] == 0:
            temp_pks_r0 = [-2, -2]
            pksi[i, j] = -2
            pksj[i, j] = -2
            pks_vals[i, j] = -2
            
        else:
            dist = np.sqrt(np.sum((temp_pks-r0)**2, axis=1))
            dist_bool = dist <= epsilon
            
            if dist_bool.any():
                temp_pks_r0 = temp_pks[dist_bool][0]
                pksi[i, j] = temp_pks_r0[0]
                pksj[i, j] = temp_pks_r0[1]
                pks_vals[i, j] = temp_cam[temp_pks_r0[0], temp_pks_r0[1], j]
                r0 = temp_pks_r0
                
                # plt.figure(1)
                # plt.imshow(temp_cam[:, :, j])
                # plt.scatter(temp_pks_r0[1], temp_pks_r0[0], c='red')
                # circ = plt.Circle((r0[1], r0[0]), radius=epsilon, color='red', fill=False)
                # ax = plt.gca()
                # ax.add_patch(circ)
                # plt.show()
                # plt.pause(0)
                # time.sleep(1.5)
                
            
            else:
                temp_pks_r0 = [-1, -1]
                pksi[i, j] = -1
                pksj[i, j] = -1
                pks_vals[i, j] = -1
                       
            del temp_pks_r0
            ri[i, j] = r0[0]
            rj[i, j] = r0[1]
    # print(pksi[i, j], pksj[i, j], pks_vals[i, j])
    # plt.figure(2)
    # plt.plot(pks_vals[i, :], '.-')
    # plt.grid()
    # plt.show()
#%
        # print(i, j)
    T[i] = (time.time()-T0)/60

print(T[-1])

#%% Use SPT to localize correlation peaks
import pandas as pd
from tqdm import tqdm

track_data = pd.read_csv('C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\Spots in tracks statistics_p3.csv')
# track_data = pd.read_csv('F:\\PhD\\E_coli\\may2021\\5\\Spots in tracks statistics.csv')


r0_track_df = track_data[track_data['TRACK_ID'] == 3]
# r0_track = r0_track_df[['POSITION_Y', 'POSITION_X']].values + [14, 13]
r0_track = r0_track_df[['POSITION_Y', 'POSITION_X']].values


pks = np.empty((number_of_images, number_of_filters), dtype=object)
pksi = np.zeros((number_of_images, number_of_filters), dtype='int')
pksj = np.zeros((number_of_images, number_of_filters), dtype='int')
pks_vals = -10*np.ones((number_of_images, number_of_filters))
# r0 = [327, 471]
# r0 = [356, 536]
# r0 = [86, 125]
# r0 = [70, 113]   # For npz array of 44x44 filter
ri = np.zeros((number_of_images, number_of_filters), dtype='int')
rj = np.zeros((number_of_images, number_of_filters), dtype='int')
epsilon = 10
T0 = time.time()
T = np.empty(number_of_images)

for i in tqdm(range(number_of_images)):
# for i in range(10):
    temp_cam = CAMERA_PHOTO[:, :, i*number_of_filters:i*number_of_filters+number_of_filters]
    # temp_cam = normalize_slices(temp_cam)
    # temp_cam = normalize_total(temp_cam)
    # temp_cam = temp_cam / temp_cam.max()
    # temp_cam, cdf = f.histeq(temp_cam)

    # Histogram equalization of every slice    
    # tt = np.empty_like(temp_cam)
    # for jj in range(temp_cam.shape[-1]):
    #     tt[:, :, jj], _ = f.histeq(temp_cam[:, :, jj])
    # temp_cam = tt

    temp_pks = np.zeros((1, 3))
    r0 = r0_track[i, :]

#%
    for j in range(number_of_filters):
#%
        # j=38
        # temp_pks= peak_local_max(temp_cam[:, :, j], num_peaks=1)
        temp_pks = peak_local_max(temp_cam[:,:,j], min_distance=25, threshold_rel=0.7)
        # temp_pks = peak_local_max(temp_cam[:,:,j], min_distance=25, threshold_abs=0.5)
        
        # print(temp_pks, np.sqrt(np.sum((temp_pks-r0)**2, axis=1)))
#%    
        # tt = np.where(temp_cam[:, :, j]==temp_cam[:, :, j].max())
        # temp_pks = np.array([[tt[0][0], tt[1][0]]])
        
        if temp_pks.shape[0] == 0:
            temp_pks_r0 = [-2, -2]
            pksi[i, j] = -2
            pksj[i, j] = -2
            pks_vals[i, j] = -2
            
        else:
            dist = np.sqrt(np.sum((temp_pks-r0)**2, axis=1))
            dist_bool = dist <= epsilon
            
            if dist_bool.any():
                temp_pks_r0 = temp_pks[dist_bool][0]
                pksi[i, j] = temp_pks_r0[0]
                pksj[i, j] = temp_pks_r0[1]
                pks_vals[i, j] = temp_cam[temp_pks_r0[0], temp_pks_r0[1], j]
                # r0 = temp_pks_r0
                
                # plt.figure(1)
                # plt.subplot(1,2,1)
                # plt.imshow(temp_cam[:, :, j])
                # plt.subplot(1,2,2)
                # plt.imshow(temp_cam[:, :, j])
                # plt.scatter(temp_pks[:, 1], temp_pks[:, 0], c='yellow', marker='H', s=70)
                # plt.scatter(temp_pks_r0[1], temp_pks_r0[0], c='red')
                # circ = plt.Circle((r0[1], r0[0]), radius=epsilon, color='red', fill=False)
                # ax = plt.gca()
                # ax.add_patch(circ)
                # plt.show()                
            
            else:
                temp_pks_r0 = [-1, -1]
                pksi[i, j] = -1
                pksj[i, j] = -1
                pks_vals[i, j] = -1
                       
            del temp_pks_r0
            ri[i, j] = r0[0]
            rj[i, j] = r0[1]
    # print(pksi[i, j], pksj[i, j], pks_vals[i, j])
    # plt.figure(2)
    # plt.plot(pks_vals[i, :], '.-')
    # plt.grid()
    # plt.show()
#%
        # print(i, j)
    T[i] = (time.time()-T0)/60

print(T[-1])


#%% Not for SPT
frame_number = np.arange(1, len(pks_vals)+1)
# pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number
max_val = np.amax(pks_vals, axis=1) 
max_val_locs = np.where((max_val != -1) & (max_val != -2))[0]  # index of rows of selected values different from -1 and -2

frame_val = frame_number[max_val_locs]
pks_vals = pks_vals[max_val_locs]
pksi = pksi[max_val_locs, :]
pksj = pksj[max_val_locs, :]

pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number

for i in range(len(pks_max)):
    if pks_max[i, :].all():
        pks_max[i, :] = False
            
image_match, filter_match = np.where(pks_max == True)
coord_i = pksi[image_match, filter_match]
coord_j = pksj[image_match, filter_match]
frame = frame_number[image_match]   

#%% For SPT

max_val = np.amax(pks_vals, axis=1) 
max_val_locs = np.where((max_val != -1) & (max_val != -2))[0]  # index of rows of selected values different from -1 and -2

coord_i = r0_track[max_val_locs, 0]
coord_j = r0_track[max_val_locs, 1]


pks_vals = pks_vals[max_val_locs]


pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number

for i in range(len(pks_max)):
    v = np.where(pks_max[i, :] == True)[0]
    if pks_max[i, :].all():
        pks_max[i, :] = False
        
    elif len(v) > 1:
        pks_max[i, v[0]] = False
            
image_match, filter_match = np.where(pks_max == True)

#%% CSAPS Smoothing
import functions as f

L = np.stack((coord_j, coord_i, filter_match), axis=1)
LL = pd.DataFrame(L, columns=['X', 'Y', 'Z'])

[x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 0.999999, True)
# [x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 1, True)

#%% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(coord_j, coord_i, filter_match, c=filter_match, label='Detected Positions')
ax.plot(x_smooth, y_smooth, z_smooth, c='red', label='Detected Positions')
# ax.plot(pos_i, pos_j, MAX_FILT, 'r-', label='Smoothed Curve')
pyplot.show()


#%% Plotly scatter plot
# import plotly.express as px
# import pandas as pd
# from plotly.offline import plot

# CURVE = pd.DataFrame(np.stack((x_smooth, y_smooth, z_smooth), axis=1), columns=['X', 'Y', 'Z'])

# fig = px.line_3d(CURVE, x='X', y='Y', z='Z')
# fig1 = px.scatter_3d(LL, x='X', y='Y', z='Z')

# fig.add_trace(fig1.data[0])

# fig.update_traces(marker=dict(size=1))
# plot(fig)

