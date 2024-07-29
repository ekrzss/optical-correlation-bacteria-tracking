# -*- coding: utf-8 -*-
"""
Created on Sat May  1 18:23:58 2021

@author: eers500
"""

import glob
import numpy as np
import matplotlib as mpl
import pandas as pd
from skimage.feature import peak_local_max
mpl.rc('figure',  figsize=(10, 6))
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f
import time
import easygui as gui
from scipy import ndimage
from numba import vectorize, jit

#%% Import Video correlate
# VID = f.videoImport("E://PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
# VID = f.videoImport("/home/erick/Documents/PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
# VID = VID[:, :, :21]
path_vid = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
VID = f.videoImport(path_vid, 0)
MAX_VID = np.max(VID)
VID = np.uint8(255*(VID / MAX_VID))
# VID = VID[:, :, :1000]
# VID = VID[:, :, 0:-1:5]
#%% Import LUT form images
#LUT = [cv2.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
# LUT = [mpimg.imread(file) for file in np.sort(glob.glob("E://PhD/23_10_19/LUT_MANUAL/*.png"))]
# LUT = [mpimg.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
# LUT = np.swapaxes(np.swapaxes(LUT, 0, 1), 1, 2)
path_lut = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
LUT = f.videoImport(path_lut, 0)
# LUT = np.uint8(255*(LUT / MAX_VID))

# LUT = VID[151-18:151+18, 162-18:162+18, :]  # Particle 1 for collodids
# LUT = VID[77-18:77+18, 332-18:332+18, :]  # P2
# LUT = VID[379-18:379+18, 130-18:130+18, :]  # P3
#LUT = VID[369-18:369+18, 292-18:292+18, :]  # P4

#%% Import 2D track
import pandas as pd
import easygui as gui

path_track = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
track_data = pd.read_csv(path_track)

# track_data = pd.read_csv('F:\PhD\Archea_LW\Spots in tracks statistics.csv')
# r0_track_df = track_data[track_data['TRACK_ID'] == 3]

# track_data = pd.read_csv('F:\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07\\Spots in tracks statistics.csv')
# r0_track_df = track_data[track_data['TRACK_ID'] == 0]

# track_data = pd.read_csv('F:\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-06\\Spots in tracks statistics.csv')
#r0_track_df = track_data[track_data['TRACK_ID'] == 0]

# r0_track_df = track_data[track_data['TRACK_ID'] == 35]
r0_track_df = track_data[track_data['TRACK_ID'] ==58]


r0_track = r0_track_df[['POSITION_Y', 'POSITION_X']].values

# plt.imshow(VID[:, :, 0])
# plt.scatter(r0_track[:, 1], r0_track[:, 0], c='red')
# circ = plt.Circle((r0_track[-1, 1], r0_track[-1,0]), radius=23, color='red', fill=False)
# ax = plt.gca()
# ax.add_patch(circ)

dr = 23 # 40 for ecoli, 23 for Archea, 90 for new ecoli
vid_surr = np.empty((2*dr, 2*dr, len(r0_track)), dtype='uint8')
for k in range(len(r0_track)):
    ii = int(r0_track[k, 0])
    jj = int(r0_track[k, 1])
    vid_surr[:, :, k] = VID[ii-dr:ii+dr, jj-dr:jj+dr, k]
    # vid_surr[:,:,k] = CAMERA_PHOTO[ii-dr:ii+dr, jj-dr:jj+dr, k]
    
f.exportAVI('vid_surr_p3.avi', vid_surr, 46, 46, 30)
# f.exportAVI('lut_p35.avi', )

#%% Create correlation array from CAMERA_PHOTOas local video result
corr = np.empty((2*dr, 2*dr, number_of_images*number_of_filters))

# for i in range(number_of_images):
#     for j in range(number_of_filters):
#         corr[:, :, i*number_of] = CAMERA_PHOTO[i-dr:i+dr, j-dr:j+dr, ]

for k in range(len(r0_track)):
    num = k*number_of_filters
    ii, jj = r0_track[k, :].astype('int')
    corr[:, :, num:num+number_of_filters] = CAMERA_PHOTO[ii-dr:ii+dr, jj-dr:jj+dr, num:num+number_of_filters] 

#%%
s = -10

plt.subplot(1,2,1)
plt.imshow(CAMERA_PHOTO[:, :, s])

plt.subplot(1,2,2)
plt.imshow(corr[:, :, s])


#%% Export vid_surr as png images
# import cv2
# import os

# vid_surr_binary = np.zeros(np.shape(vid_surr))
# vid_surr_binary[vid_surr >= np.mean(vid_surr)]  = 255

# p = "F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\local_video_p35\\"
# for i in range(vid_surr.shape[-1]):
# # for i in range(20):
#     # cv2.imwrite(os.path.join(p, '{0:2d}.png'.format(i)), vid_surr_binary[:, :, i])
#     # name = p+'{0:2d}.png'.format(i)
#     name = p+"im_{0:2d}.png".format(i)
#     plt.imsave(name, vid_surr_binary[:,:,i], cmap='gray')
    
#%% Prepare arrays
LUT_BINARY = np.zeros(np.shape(LUT))
VID_BINARY = np.zeros(np.shape(vid_surr))

LUT_BINARY[LUT >= np.mean(LUT)] = 255
VID_BINARY[vid_surr >= np.mean(vid_surr)] = 255

#CORR = np.empty((np.shape(VID)[0], np.shape(VID)[1] , np.shape(VID)[2] * np.shape(LUT)[2]), dtype='float32')

A = np.repeat(VID_BINARY.astype('uint8'), repeats=LUT.shape[-1], axis=-1).astype('float16')
B = np.tile(LUT_BINARY, VID.shape[-1]).astype('float16')

# del VID, VID_BINARY, LUT, LUT_BINARY

#%% Correltion in GPU
#@vectorize(["complex128(complex128, complex128)"], target='cuda')   #not good
@jit(nopython=True) 
def corr_gpu(a, b):
    return a*np.conj(b)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

BB = np.empty_like(A)
for k in range(np.shape(B)[2]):
    # BB[:, :, k] = np.pad(B[:, :, k], int((1024-110)/2))
    BB[:, :, k] = np.pad(B[:, :, k], int((A.shape[0]-B.shape[0])/2))      #A.shape[0] should be bigger than B.shape
del B
FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

C = np.empty_like(A, dtype='float32')
T0 = time.time()
T = []
for k in range(np.shape(A)[2]):
# for k in range(100):
    print(k)
    BFT = FT(BB[:,:,k])
    R = corr_gpu(FT(A[:, :, k]), BFT).astype('complex64')
    # C[:, :, k] = np.abs(IFT(R))
    C[:, :, k] = np.abs(IFT(R / np.sum(BFT)))    # Normalize with sum of pixel values of filters
    # C[:, :, k] = np.abs(IFT(R))
    T.append((time.time()-T0)/60)
print(T[-1])

# del A, BB
# CC = np.reshape(C, (226*39000, 226))
# np.savetxt('F:\PhD\Archea_LW\LUT_CES_30\GPU_corr_21.58min_226.txt', CC, fmt='%i', delimiter=',')

# np.savez_compressed('F:\PhD\Archea_LW\LUT_CES_44\GPU_corr_7.78min_226_400frames_every5_normalised_filter_sum_squared.npz', a=C)
# 22 seconds

#%%
number_of_images = 275
number_of_filters = 30

CAMERA_PHOTO, cdf = f.histeq(C)
pks = np.empty((number_of_images, number_of_filters), dtype=object)
pksi = np.zeros((number_of_images, number_of_filters), dtype='int')
pksj = np.zeros((number_of_images, number_of_filters), dtype='int')
pks_vals = -10*np.ones((number_of_images, number_of_filters))
# Archea
# r0 = [327, 471]
# r0 = [356, 536]
# r0 = [86, 125]
# r0 = [70, 113]   # For npz array of 44x44 filter
# r0 = [187, 35]     # For particle 2

# Ecoli
r0 = [40, 40]
ri = np.zeros((number_of_images, number_of_filters), dtype='int')
rj = np.zeros((number_of_images, number_of_filters), dtype='int')
epsilon = 5
T0 = time.time()
T = np.empty(number_of_images)

for i in range(number_of_images):
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
                # r0 = temp_pks_r0
                
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
        print(i, j)
    T[i] = (time.time()-T0)/60

print(T[-1])

#%%
# frame_number = np.arange(1, len(pks_vals)+1)
# pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number
max_val = np.amax(pks_vals, axis=1) 
max_val_locs = np.where((max_val != -1) & (max_val != -2))[0]  # index of rows of selected values different from -1 and -2

coord_i = r0_track[max_val_locs, 0]
coord_j = r0_track[max_val_locs, 1]

# frame_val = frame_number[max_val_locs]
pks_vals = pks_vals[max_val_locs]
# pksi = pksi[max_val_locs, :]
# pksj = pksj[max_val_locs, :]

pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number

for i in range(len(pks_max)):
    if pks_max[i, :].all():
        pks_max[i, :] = False
            
image_match, filter_match = np.where(pks_max == True)
# coord_i = pksi[image_match, filter_match]
# coord_j = pksj[image_match, filter_match]
# frame = frame_number[image_match]   



#% CSAPS Smoothing
import functions as f

L = np.stack((coord_j, coord_i, filter_match), axis=1)
LL = pd.DataFrame(L, columns=['X', 'Y', 'Z'])

[x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 0.999999, True)
# [x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 1, True)

#% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coord_j, coord_i, filter_match, c=filter_match, label='Detected Positions')
ax.plot(x_smooth, y_smooth, z_smooth, c='red', label='Detected Positions')
# ax.plot(pos_i, pos_j, MAX_FILT, 'r-', label='Smoothed Curve')
pyplot.show()

#%%

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
ax[0, 0].imshow(VID_BINARY[:, :, 0])
ax[0, 1].imshow(LUT_BINARY[:, :, 3])
ax[0, 2].imshow(LUT_BINARY[:, :, 22])
ax[1, 1].imshow(C[:, :, 0])
ax[1, 2].imshow(C[:, :, 22])

plt.figure(2)
plt.plot(np.max(C[:, :, :29], axis=(0,1)))

#%% Gauss fit analysis
# def peak_gauss_fit_analysis(normalized_input, peak_number, peak_array, sel_size):
def peak_gauss_fit_analysis(input2darray):
    # k = peak_number  # Peak number
    # sel_size = 15
    # DATA = normalized_input[peak_array[k][0]-sel_size:peak_array[k][0]+sel_size, peak_array[k][1]-sel_size:peak_array[k][1]+sel_size]

    sel_size = 20
    pks = peak_local_max(input2darray, num_peaks=1)
    INTENSITY = input2darray[pks[0][0], pks[0][1]]
    DATA = input2darray[pks[0][0]-sel_size:pks[0][0]+sel_size+1, pks[0][1]-sel_size:pks[0][1]+sel_size+1]
    
    if DATA.shape == (0, 0):
        print('Peak not in center')
        return
    else:
        pks = peak_local_max(DATA, num_peaks=1)
        
        def gauss(x, x0, y, y0, sigma, MAX):
            # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
            return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))    
        
        I, J = np.meshgrid(np.arange(DATA.shape[0]), np.arange(DATA.shape[1]))
        sig = np.linspace(0.1, 2, 200)
        chisq = np.empty_like(sig)
        
        for ii in range(np.shape(sig)[0]):
                chisq[ii] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], np.max(DATA)))**2)/np.var(DATA)
                    
        LOC_MIN = np.where(chisq == np.min(chisq))
        SIGMA_OPT = sig[LOC_MIN[0][0]]
        fitted_gaussian = gauss(I, pks[0][1], J, pks[0][0], SIGMA_OPT, INTENSITY) #ZZ
        OP = np.sum(DATA)
        
        return INTENSITY, SIGMA_OPT, OP, fitted_gaussian, DATA

#%%
from scipy.ndimage import gaussian_filter

data_filtered = gaussian_filter(C[:, :, 0], sigma=1)


intensity, sigma, op, gaussian, data = peak_gauss_fit_analysis(C[:, :,6])
# a = peak_gauss_fit_analysis(C[:, :,20])


_, _, _, gaussian_filtered, _ = peak_gauss_fit_analysis(data_filtered)
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# def gauss(x, x0, y, y0, sigma, MAX):
#         # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
#         return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    
fig2 = plt.figure(figsize=(12,6))
ax1 = fig2.add_subplot(221)
ax2 = fig2.add_subplot(222)
ax3 = fig2.add_subplot(223, projection='3d')
ax4 = fig2.add_subplot(224, projection='3d')

# gaussian = gauss(I, pks[0][1], J, pks[0][0], 3, INTENSITY)

x = np.arange(len(gaussian))
y = np.arange(len(gaussian))
X,Y = np.meshgrid(x,y)

ax1.imshow(C[:, :, 20])
# ax2.imshow(fitted_gaussian)
ax2.imshow(gaussian)


# Plot a basic wireframe
ax3.plot_surface(X, Y, data)
# ax1.set_title('row step size 10, column step size 10')


# ax4.plot_surface(X, Y, fitted_gaussian)
ax4.plot_surface(X, Y, gaussian)
# ax2.set_title('row step size 20, column step size 20')


plt.show()

#%%
# from scipy.optimize import 
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter, mean

input_array = C[:, :, 20]
input_array = gaussian_filter(input_array, sigma=1)

max_value = np.max(input_array)
pki, pkj = np.where(input_array == max_value)
pki, pkj = pki[0], pkj[0]

def gauss(x, x0, y, y0, sigma, MAX, H):
        return H + MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))   
    
I, J = np.meshgrid(np.arange(len(input_array)), np.arange(len(input_array)))
sigma = np.linspace(0.1, 20, 100)
H = np.linspace(np.min(input_array), np.max(input_array), 100)
var = np.var(input_array)

chisq = np.empty((len(sigma), len(H)))
for ii in range(len(sigma)):
    for jj in range(len(H)):
        chisq[ii, jj] = np.sum((input_array - gauss(I, pki, J, pkj, sigma[ii], max_value, H[jj]))**2) / var

mini, minj = np.where(chisq == chisq.min())
mini, minj = mini[0], minj[0]

sigma_min = sigma[mini]
H_min = H[minj]

gauss_fit = gauss(I, pki, J, pkj, sigma_min, max_value, H_min)


s, h = np.meshgrid(sigma, H)
plt.figure(1)         
plt.contour(s, h, chisq, levels=800)
# plt.scatter(H_min, sigma_min, c='red')
    
fig2 = plt.figure(figsize=(12,6))
ax1 = fig2.add_subplot(221)
ax2 = fig2.add_subplot(222)
ax3 = fig2.add_subplot(223, projection='3d')
ax4 = fig2.add_subplot(224, projection='3d')

# gaussian = gauss(I, pks[0][1], J, pks[0][0], 3, INTENSITY)

x = np.arange(len(gauss_fit))
y = np.arange(len(gauss_fit))
X,Y = np.meshgrid(x,y)

ax1.imshow(input_array)
# ax2.imshow(fitted_gaussian)
ax2.imshow(gauss_fit)
ax2.set_title('sigma = '+np.str(sigma_min.astype('float16'))+' H = '+np.str(H_min.astype('float32')))


# Plot a basic wireframe
ax3.plot_surface(X, Y, input_array)
# ax1.set_title('row step size 10, column step size 10')


# ax4.plot_surface(X, Y, fitted_gaussian)
ax4.plot_surface(X, Y, gauss_fit)
# ax2.set_title('row step size 20, column step size 20')


plt.show()














































