# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:32:39 2021

@author: eers500
"""

import numpy as np
import cupy as cp
import matplotlib as mpl
mpl.rc('figure',  figsize=(5, 5))
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import functions as f
import easygui as gui
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import time
from tqdm import tqdm
from numba import vectorize, jit

#%% Import Video correlate
path_vid = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
VID = f.videoImport(path_vid, 0)
ni, nj, nk = np.shape(VID)

invert = True

if invert:
    for i in range(nk):
        VID[:, :, i] = VID[:, :, i].max() - VID[:, :, i]
        
#%% Remove background by using moving average
# nn = 5
# vid = np.empty_like(VID)
# for kk in tqdm(range(nk)):
    
#     if kk in list(np.arange(nn+1)):
#         MED = np.median(VID[:, :, :2*nn], axis=2)
#         vid[:, :, kk] = VID[:, :, kk] - MED
#         vid[:, :, kk] = vid[:, :, kk].max() - vid[:, :, kk]
    
#     elif kk in list(np.arange(nk-nn+1, nk)):
#         MED = np.median(VID[:, :, nk-nn+1:nk], axis=2)
#         vid[:, :, kk] = VID[:, :, kk] - MED
#         vid[:, :, kk] = vid[:, :, kk].max() - vid[:, :, kk]

#     else:
#         MED = np.median(VID[:, :, kk-nn:kk+nn], axis=2)
#         vid[:, :, kk] = VID[:, :, kk] - MED
#         vid[:, :, kk] = vid[:, :, kk].max() - vid[:, :, kk]
    

#%% Import LUT form images
path_lut = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
LUT = f.videoImport(path_lut, 0)
mi, mj, mk = np.shape(LUT)

if invert:
    for i in range(mk):
        LUT[:, :, i] = LUT[:, :, i].max() - LUT[:, :, i]

#%% Prepare arrays
VID_zn = np.empty_like(VID)
for k in range(nk):
    A = VID[:,:,k]
    VID_zn[:, :, k] = (A-np.mean(A))/np.std(A)

LUT_zn = np.empty_like(LUT)
for k in range(mk):
    A = LUT[:,:,k]
    LUT_zn[:, :, k] = (A-np.mean(A))/np.std(A)

#%% Correltion in GPU
#@vectorize(["complex128(complex128, complex128)"], target='cuda')   #not good
# @jit(nopython=True) 
# def corr_gpu(a, b):
#     return a*np.conj(b)

# def pad_with(vector, pad_width, iaxis, kwargs):
#     pad_value = kwargs.get('padder', 0)
#     vector[:pad_width[0]] = pad_value
#     vector[-pad_width[1]:] = pad_value


# FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
# IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

# CC = np.empty((ni, nj, nk*mk))
# T0 = time.time()
# T_CORR = []
# for i in tqdm(range(nk)):
# # for i in range(10):
#     im = VID_zn[:, :, i]
#     imft = FT(im)
#     for j in range(mk):
#         fm = np.pad(LUT_zn[:, :, j], int((ni-mi)/2))
#         fmft = FT(fm)
#         CC[:, :, i*mk+j] = np.abs(IFT(corr_gpu(imft, fmft)))
#         T_CORR.append((time.time()-T0)/60)
# print(T_CORR[-1])

#%% CuPy correlation
def corr_gpu(a, b):
    return a*cp.conj

cFT = lambda x: cp.fft.fftshift(cp.fft.fft2(x))
cIFT = lambda X: cp.fft.ifftshift(cp.fft.ifft2(X))

CC = np.empty((ni, nj, nk*mk), dtype='float32')
T0 = time.time()
T_CORR = []
for i in tqdm(range(nk)):
# for i in range(10):
    im = VID_zn[:, :, i]
    imft = cFT(cp.array(im))
    for j in range(mk):
        fm = cp.pad(cp.array(LUT_zn[:, :, j]), int((ni-mi)/2))
        fmft = cFT(fm)
        # CC[:, :, i*mk+j] = np.abs(cIFT(corr_gpu(imft, fmft)))
        CC[:, :, i*mk+j] = cp.abs(cIFT(imft*cp.conj(fmft))).get()
        T_CORR.append((time.time()-T0)/60)
print(T_CORR[-1])


# paths = ['C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\ZNCC\\CC_CuPy_32f.npy',
#          'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\CC_CuPy_32f.npy',
#          'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\CC_CuPy_32f.npy',
#          'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\CC_CuPy_32f.npy',
#          'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1_07\\CC_CuPy_32f.npy']


# CC = np.load(paths[1])

#%% Video settings
magnification = 20          # Archea: 20, E. coli: 40, 40, 40, MAY 20
frame_rate = 100              # Archea: 30/5, E. coli: 60, 60, 60, MAY 100
fs = 0.711                  # px/um
SZ = 20                     # step size of LUT [Archea: 10um,E. coli: 20, 40, 20, MAY 20]
number_of_images = nk      # Archea = 400 , Ecoli = 430, 430, 700  # MAY 275(550)
number_of_filters = mk      # Archea =  25 ,   Ecoli =  19,  19,  20  # MAY 30  


#%% Use SPT 2D trajectory to obtain 3D coordinates of correlation peaks
# pnumber = 3
# path_track = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
# track_data = pd.read_csv(path_track)
# r0_track_df = track_data[track_data['TRACK_ID'] == pnumber]
# r0_track = r0_track_df[['POSITION_Y', 'POSITION_X']].values

eps_gauss = 1                 # Archea: [0.2, 1] , E. coli: 0.5, 0.7, 1, MAY 1
threshold_pcg = 0.3         # Archea: 0.3  E. coli: 0.25, -, 0.15, MAY 0.3

VID_filtered = np.empty_like(VID)
for k in tqdm(range(nk)):
    im, _ = np.real(f.bandpassFilter(VID[:, :, k], 2, 30))
    imm = np.zeros_like(im) 
    imm[im > threshold_pcg*im.max()] = 255
    VID_filtered[:, :, k] = gaussian_filter(imm, eps_gauss)
    
#%
n=0
r = peak_local_max(VID_filtered[:, :, n], threshold_rel=threshold_pcg, min_distance=20) # Archea 20, Ecoli 30, 20, 20, MAY 20

# plt.imshow(VID_filtered[:, :, 0])
# plt.scatter(r[:, 1], r[:, 0], facecolors='none', edgecolors='r')

#%% 2D Tracking
tracks_2d = []
vids = []

st1, st2 = [], []

T_2D_track = []
T0 = time.time()
print('Performing 2D tracking ...')
for i, p in enumerate(tqdm(r)):        
    r0 = f.track_2d(VID_filtered, p, 20, threshold_pcg)
    stdi = np.std(r0[:, 0])
    stdj = np.std(r0[:, 1])
    st1.append(stdi)
    st2.append(stdj)
    
    if stdi > 5 or stdj > 5:                # Archea 5, E. coli: -, 5, 5, MAY 5
        # tracks[i] = r0
        # vids[i] = VID_filtered
        tracks_2d.append(r0)
        vids.append(VID_filtered)
        
    # tracks[i] = r0
    # vids.append(VID_filtered)
    # tracks_2d.append(r0)
    T_2D_track.append((time.time()-T0)/60)
# tracks_2d = list(tracks) 

# tracks_2d = []
# st1, st2 = [], []
# for t in t_2d:
#     stdi = np.std(t[:, 0])
#     stdj = np.std(t[:, 1])
#     st1.append(stdi)
#     st2.append(stdj)
#     if stdi > 7 or stdj > 7:
#         tracks_2d.append(t)
#%
# for i in range(len(tracks_2d)):
    # i=18
    # plt.plot(tracks_2d[i][:,1], -tracks_2d[i][:, 0], '.')

#%% Remove static noise in tracks to drop short ones

id_drop = []
for k in range(len(tracks_2d)):
    ft = tracks_2d[k]
    frame= np.arange(len(ft))
    
    fft = ft[:-2,:] 
    diffi = np.diff(ft[:, 0])
    diffj = np.diff(ft[:, 1])
    
    ftfi = f.contiguous_repeats(diffi)
    ftfj = f.contiguous_repeats(diffj)
    
    ftfi_f = np.where(ftfi > 15)[0]
    ftfj_f = np.where(ftfj > 15)[0]
    
    ids  = np.unique(np.concatenate((ftfi_f, ftfj_f)))
    idsb = np.array([i for i in frame if i not in ids])
    id_drop.append(idsb)

    

# plt.subplot(3,1,1)
# plt.plot(frame, ft[:, 0], '.-'); plt.plot(frame, ft[:, 1], '.-')
# plt.subplot(3,1,2)
# plt.plot(frame[idsb], ft[idsb, 0], '.-'); plt.plot(frame[idsb], ft[idsb, 1], '.-')
# plt.xlim(frame[0], frame[-1])
# plt.subplot(3,1,3)
# plt.plot(ftfi); plt.plot(ftfj)
# plt.show()

#%% Obtain Z-coordinate
N = CC.shape[-1]
step = 3
method = 'std_dev' # max or std_dev
ds = 15
filter_match = []

T_3D_track = []
T0 = time.time()
print('Creating 3D tracks...')
for tnum, track in enumerate(tqdm(tracks_2d)):
    r0 = track    
    std_dev = np.nan*np.ones((number_of_images, number_of_filters))
    max_val = np.nan*np.ones_like(std_dev)
    fit = np.empty((number_of_images, number_of_filters), dtype='object')
    data = np.empty((number_of_images, number_of_filters), dtype='object')
    f_match = np.nan*np.ones(number_of_images)
    
    for i in range(number_of_images):
    # for i in range(38):
        
        if i==0:
            for j in range(number_of_filters):
                # temp_corr = CC[:, :, i*number_of_filters+j]
                # j=18
                temp_corr = CC[r0[i][0]-ds:r0[i][0]+ds,r0[i][1]-ds:r0[i][1]+ds , i*number_of_filters+j]
                # plt.imshow(temp_corr)
                
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
                    f_match[i] = np.nan
                elif not np.isnan(std_dev[i, :]).all(): 
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
                # temp_corr = CC[:, :, i*number_of_filters+j]
                temp_corr = CC[r0[i][0]-ds:r0[i][0]+ds,r0[i][1]-ds:r0[i][1]+ds , i*number_of_filters+j]
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
                    if np.isnan(f_match[i-1]):
                        continue
                    else:
                        f_match[i] = f_match[i-1]
                else:
                    f_match[i] = np.where(std_dev[i, :] == np.nanmin(std_dev[i, :]))[0][0]
            elif method == 'max':
                f_match[i] = np.where(max_val[i, :] == np.nanmax(max_val[i, :]))[0][0]
     
    filter_match.append(f_match)
    T_3D_track.append((time.time() - T0)/60)

print(T_3D_track[-1]/60)

#%% Create tracks_3d Dataframe
zz = np.arange(number_of_filters-1, -1, -1)*SZ   
    
tracks_3d  = -np.ones((1, 5))
for i, t in enumerate(tracks_2d):
    
    real_z = np.empty_like(filter_match[i])
    
    for k in range(len(filter_match[i])):
        # filter_match = zz[int(filt)]
        real_z[k] = zz[int(filter_match[i][k])]
                       
    tt = np.hstack((t*(1/fs)/magnification, np.expand_dims(filter_match[i], axis=1)))
    tt = np.hstack((tt, np.expand_dims(real_z, axis=1)))
    tt = np.hstack((tt, i*np.ones((len(t), 1))))
    tracks_3d = np.vstack((tracks_3d, tt))
    
tracks_3d = pd.DataFrame(tracks_3d[1:, :], columns=['Y', 'X', 'FILTER', 'Z', 'ID'])
tracks_3d['Frame'] = np.tile(np.arange(len(tracks_2d[0])), len(tracks_2d))

dframes = []
for k in range(len(id_drop)):
    dataframe = tracks_3d[tracks_3d['ID'] == k]
    dataframe = dataframe.iloc[id_drop[k]]
    if len(dataframe) > 100:
        dframes.append(dataframe)

tracks_3d = pd.concat(dframes)
tracks_3d['T'] = (1/frame_rate)*tracks_3d['Frame']
    
#%% 3D scatter 1 curve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

# tnum = 12
# curve = tracks_3d[tracks_3d['ID'] == tnum]
# ax1.scatter(curve.Y, curve.X, curve.Z, s=5, c=curve.Frame)
# ax1.scatter(curve.Y.values[idsb], curve.X.values[idsb], curve.Z.values[idsb], s=5, c=curve.Frame.values[idsb])    

ax1.scatter(tracks_3d.Y, tracks_3d.X, tracks_3d.Z, c=tracks_3d.Frame, s=5)
# ax1.plot(tracks_3d.Y, tracks_3d.X, tracks_3d.Z)

pyplot.show()


#%% 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# fig = plt.figure(1)
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.scatter(idmin[:, 0], idmin[:, 1], -idmin[:, 4], '.', c=idmin[:, 3])
# # ax1.scatter(idmax[:, 0], idmax[:, 1], idmax[:, 4], '.', c=idmax[:, 3])
# pyplot.show() 
    

#%% CSAPS Smoothing
# import functions as f

spline_degree = 3  # 3 for cubic spline
particle_num = np.unique(tracks_3d.ID)
T0_smooth = time.time()

dframes = []
for pn in particle_num:
    # Do not use this
    # L = LINKED[LINKED.PARTICLE == pn].values
    # X = f.smooth_curve(L, spline_degree=spline_degree, lim=20, sc=3000)
    
    L = tracks_3d[tracks_3d.ID == pn]
    if len(L) < 100:
        continue
    X = f.csaps_smoothing(L, smoothing_condition=0.99999, filter_data=False)
    
    if X != -1:
        X.append(pn*np.ones_like(X[1]))
        sc = pd.DataFrame(X, ['X', 'Y', 'Z', 'T', 'ID'])
        # smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], X[3], pn*np.ones_like(X[1])), axis=1))) 
        dframes.append(pd.DataFrame.transpose(sc))
        
smoothed_curves_df = pd.concat(dframes)
        
T_smooth = time.time() - T0_smooth

#%% 3D Line Plot smooth
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt
plt.rcParams['figure.dpi'] = 150 
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['ID'] == pn]
    ax.scatter(s.Y, s.X, s.Z, 'r')
    ax.plot(s.Y, s.X, s.Z, 'r', linewidth=1)
ax.axis('tight')
ax.set_xlabel('x ($\mu$m)')
ax.set_ylabel('y ($\mu$m)')
ax.set_zlabel('-z ($\mu$m)')
# ax.scatter(coord_j, coord_i, filter_match, c=frame, label='Detected Positions')
# ax.plot(smoothed_curves_df.X, smoothed_curves_df.Y, smoothed_curves_df.X, label='Detected Positions')
# ax.plot(pos_i, pos_j, MAX_FILT, 'r-', label='Smoothed Curve')
pyplot.show()


#%% Plotly scatter plot
# import plotly.express as px
# import pandas as pd
# from plotly.offline import plot

# 
# CURVE = smoothed_curves_df
# # CURVE = smoothed_curves_df[smoothed_curves_df['ID'] == 0]

# fig = px.line_3d(CURVE, x='Y', y='X', z='Z', color='ID')
# # fig = px.scatter_3d(CURVE, x='X', y='Y', z='Z', color='ID')


# # fig['layout']['scene']['aspectmode'] =  'cube'
# fig.update_traces(marker=dict(size=1))
# plot(fig)

