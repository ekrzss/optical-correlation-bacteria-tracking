# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:14:56 2021

@author: eers500
"""
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time
import easygui as gui
import pandas as pd
from skimage import restoration
from skimage.feature import peak_local_max
from tqdm import tqdm

type_of_correlation = 'GPU' # CPU, GPU or Optical

#%% Import 2D track and load correaltion array

magnification = 20
fs = 0.711                  # px/um
ps = (1/fs)/magnification
SZ = 20                     # step size of LUT [Archea: 10um,E. coli: 20, 40, 20, MAY 20]
pnumber = 0                # Archea: 3, E.coli 35, 58, 4   MAY 0
path_track = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
track_data = pd.read_csv(path_track)
r0_track_df = track_data[track_data['TRACK_ID'] == pnumber]
r0_track = r0_track_df[['POSITION_Y', 'POSITION_X']].values

# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\Octypus_batch\\'+'')   # Archea - particle 3
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\'+'CC.npy')   # Ecoli Sample 1 - 03 - particle 35
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\'+'CC.npy')   # Ecoli Sample 1 - 06 - particle 58
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\'+'CC.npy')   # Ecoli Sample 2 - particle 4
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\31_aug_21\\CC_4x.npy') # Sample 2 - particle 4
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07\\'+'CC.npy') #particle 0

# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\ZNCC\\GPU\\C_corr_zn.npy')
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\ZNCC\\GPU\\C_corr_zn.npy') # - particle 35
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\ZNCC\\GPU\\C_corr_zn.npy')   # - particle 58
# CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\ZNCC\\GPU\\C_corr_zn.npy')   # Ecoli Sample 2 - particle 4

#%%
# def gauss(x, x0, y, y0, sigma, MAX):
#            # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
#            return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

# # def peak_gauss_fit_analysis(normalized_input, peak_number, peak_array, sel_size):
# def peak_gauss_fit_analysis(input2darray):
#     # k = peak_number  # Peak number
#     # sel_size = 15
#     # DATA = normalized_input[peak_array[k][0]-sel_size:peak_array[k][0]+sel_size, peak_array[k][1]-sel_size:peak_array[k][1]+sel_size]

#     # sel_size = 10
    
#     si, sj = input2darray.shape
#     dr = 20  #20 50
#     sel_size = dr
#     rmid = [int(si/2), int(sj/2)]
    
#     # T0 = time.time()
#     temp_input = input2darray[int(si/2)-dr:int(si/2)+dr, int(sj/2)-dr:int(sj/2)+dr]
#     pkss = peak_local_max(temp_input, num_peaks=10, threshold_rel=0.8)   
#     pks = (rmid+pkss)-dr
    
#     if len(pks) == 0:
#         return 'Empty'
#     # print(time.time()-T0)
    
#     # T0 = time.time()
#     # pkss = peak_local_max(input2darray, num_peaks=10)
#     # print(time.time()-T0)
#     dist_to_rmid = np.sqrt(np.sum((pks-rmid)**2, axis=1))
#     # pks_near_rmid = pks[dist_to_rmid <= dr, :]
#     pks_near_rmid = pks[dist_to_rmid == dist_to_rmid.min(), :]
    
#     # plt.imshow(input2darray)
#     # plt.scatter(r1[:,1], r1[:,0], c='red')
#     # plt.scatter(pks[:,1], pks[:,0], c='yellow')
    
#     if len(pks_near_rmid) == 0:
#         pks = np.array([np.array(rmid)])
#     else: 
    
#         intensity_pks_near_rmid = np.empty(len(pks_near_rmid))
#         for i in range(len(pks_near_rmid)):
#             intensity_pks_near_rmid[i] = input2darray[pks_near_rmid[i][0], pks_near_rmid[i][1]]
    
#         pks = pks_near_rmid[intensity_pks_near_rmid == intensity_pks_near_rmid.max()]
#         # dist_to_pks = np.sqrt(np.sum((pks-rmid)**2, axis=1))
    
#         if len(pks) > 1:
#             pks = np.array([pks[0]])
    

#     INTENSITY = input2darray[pks[0][0], pks[0][1]]
#     DATA = input2darray[pks[0][0]-sel_size:pks[0][0]+sel_size+1, pks[0][1]-sel_size:pks[0][1]+sel_size+1]  # centered in pks
    
#     if DATA.shape != (2*sel_size+1, 2*sel_size+1):
#         return 'Empty'

#     else:  
        
#         centeri, centerj = int(np.floor(DATA.shape[0]/2)), int(np.floor(DATA.shape[1]/2))
#         center_value = DATA[centeri, centerj]
#         I, J = np.meshgrid(np.arange(DATA.shape[0]), np.arange(DATA.shape[1]))
#         sig = np.linspace(0.1, 40, 200)
#         chisq = np.empty_like(sig)
        
#         for ii in range(len(sig)):
#             # chisq[ii] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], DATA.max()))**2)/np.var(DATA)
#             chisq[ii] = np.sum((DATA - gauss(I, centeri, J, centerj, sig[ii], center_value))**2)/np.var(DATA)
#             # for jj in range(len(MAX)):
#                 # chisq[ii, jj] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], MAX[jj]))**2)/np.var(DATA)
                    
#         LOC_MIN = np.where(chisq == np.min(chisq))
#         SIGMA_OPT = sig[LOC_MIN[0][0]]
#         # MAX_OPT = MAX[LOC_MIN[1][0]]
#         fitted_gaussian = gauss(I, centeri, J, centerj, SIGMA_OPT, INTENSITY) #ZZ
#         OP = np.sum(DATA)
        
#         # plt.figure(1)
#         # plt.plot(sig, chisq, '.-')
#         # plt.figure(2)
#         # plt.subplot(1, 2, 1); plt.imshow(DATA)
#         # plt.subplot(1, 2, 2); plt.imshow(fitted_gaussian)
        
#         return INTENSITY, SIGMA_OPT, OP, fitted_gaussian, DATA


#%%
number_of_images = 275   # Archea = 400 , Ecoli = 430, 430, 700  # MAY 275
number_of_filters = 30  #Archea =  25 ,   Ecoli =  19,  19,  20  # MAY 30  
N = CC.shape[-1]
std_dev = np.nan*np.ones((number_of_images, number_of_filters))
max_val = np.nan*np.ones_like(std_dev)
fit = np.empty((number_of_images, number_of_filters), dtype='object')
data = np.empty((number_of_images, number_of_filters), dtype='object')
f_match = np.nan*np.ones(number_of_images)
step = 1
method = 'std_dev' # max or std_dev

T = []
T0 = time.time()
for i in tqdm(range(number_of_images)):
    
    if i==0:
        for j in range(number_of_filters):
            temp_corr = CC[:, :, i*number_of_filters+j]
            vals = f.peak_gauss_fit_analysis(temp_corr)
            if vals == 'Empty':
                std_dev[i, j] = np.nan
                max_val[i, j] = np.nan
                fit[i, j] = np.nan
                data[i, j] = np.nan
            else:
                std_dev[i, j] = vals[1]
                max_val[i, j] = vals[0]
                fit[i, j] = vals[3]
                data[i, j] = vals[4]
        
        if method == 'std_dev':
            f_match[i] = np.where(std_dev[i, :] == np.nanmin(std_dev[i, :]))[0][0]
        elif method == 'max':
            f_match[i] = np.where(max_val[i, :] == np.nanmax(max_val[i, :]))[0][0]
        
    else:
        center_index = int(f_match[i-1])                                       # From previous iteration
        indices = np.arange(center_index-step, center_index+step+1)                 # Indices to to use to fit gaussian
        
        if (indices < 0).any():                                                     # Border handling
            indices_bool = indices < 0
            indices = indices[~indices_bool]
            
        elif (indices > number_of_filters-1).any():
            indices_bool = indices > number_of_filters-1
            indices = indices[~indices_bool]
        
        for j in indices:
            temp_corr = CC[:, :, i*number_of_filters+j]
            vals = f.peak_gauss_fit_analysis(temp_corr)
            if vals == 'Empty':
                std_dev[i, j] = np.nan
                max_val[i, j] = np.nan
                fit[i, j] = np.nan
                data[i, j] = np.nan
            else:
                std_dev[i, j] = vals[1]
                max_val[i, j] = vals[0]
                fit[i, j] = vals[3]
                data[i, j] = vals[4]
            
        if method == 'std_dev':
            if np.isnan(std_dev[i, :]).all():
                f_match[i] = f_match[i-1]
            else:
                f_match[i] = np.where(std_dev[i, :] == np.nanmin(std_dev[i, :]))[0][0]
        elif method == 'max':
            f_match[i] = np.where(max_val[i, :] == np.nanmax(max_val[i, :]))[0][0]
            
        T.append(time.time() - T0)
print(T[-1]/60)

#%%
n = 0
m = 0
fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].imshow(data[n, m])
ax[1].imshow(fit[n, m])
f.surf(data[n, m])
f.surf(fit[n, m])

#%%
fig, ax = plt.subplots(4, 4)
fig1, ax1 = plt.subplots(4, 4)
frame = 0
p=1

for i in range(4):
    for j in range(4):
        ax[i, j].imshow(data[frame, i*p+j])
        ax[i, j].axis('off')
        
        ax1[i, j].imshow(fit[frame, i*p+j])
        ax1[i, j].axis('off')
plt.show()

#%% Filter filter_match selection to avoid suddent jumps
from scipy import ndimage

coord_i = r0_track[:, 0]*ps
coord_j = r0_track[:, 1]*ps

zz = np.arange(number_of_filters-1, -1, -1)*SZ
    
coord_k = np.empty_like(f_match)
for k in range(len(f_match)):
    # filter_match = zz[int(filt)]
    coord_k[k] = zz[int(f_match[k])]



num = np.arange(len(coord_k)-1)
jump = np.abs(np.diff(coord_k)) 
smooth_jump = ndimage.gaussian_filter1d(jump, 1, mode='mirror')  # window of size 5 is arbitrary

# plt.figure(1)
# plt.plot(50+jump, '.-') 
# plt.plot(smooth_jump, '.-')

limit = 100*np.mean(smooth_jump)    # factor 2 is arbitrary
# limit=15

coord_kk = coord_k[:-1]
boolean = (jump >= 0) & (smooth_jump < limit)
filtered = coord_kk[boolean]

# plt.figure(2)
# plt.plot(np.arange(len(filter_match)), 50+filter_match, '.-')
# plt.plot(num[boolean], filter_sel[boolean], '.-')

coord_ii = coord_i[:-1]
coord_jj = coord_j[:-1]

coord_ii = coord_ii[boolean]
coord_jj = coord_jj[boolean]
frame = num[boolean]

# zz = np.arange(number_of_filters-1, -1, -1)*SZ
# coord_z = []
# for filt in filtered:
#     coord_z.append(zz[int(filt)])


#%
#% CSAPS Smoothing
import functions as f

# L = np.stack((coord_j, coord_i, filter_match), axis=1)
# L = np.stack((coord_jj, coord_ii, filtered), axis=1)
L = np.stack((coord_jj, coord_ii, coord_kk, frame), axis=1)
LL = pd.DataFrame(L, columns=['X', 'Y', 'Z', 'T'])

[x_smooth, y_smooth, z_smooth, t_smooth] = f.csaps_smoothing(LL, 1-1E-7, False)

#% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
# ax1.scatter(coord_i, coord_j, coord_k, c=np.linspace(0, 1, len(f_match)), label='Detected Positions')
ax1.scatter(coord_ii, coord_jj, coord_kk, c=frame, label='Detected Positions')
ax1.plot(y_smooth, x_smooth, z_smooth, c='red', label='Detected Positions')


ax1.set_xlabel('Y [$\mu$m]')
ax1.set_ylabel('X [$\mu$m]')
ax1.set_zlabel('Z[$\mu$m]')
ax1.set_title(type_of_correlation+' Correlation', fontsize=20)
pyplot.show()
        





