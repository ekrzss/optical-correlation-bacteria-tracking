# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 23:18:42 2021

@author: eers500
"""

import numpy as np
import cupy as cp
import matplotlib as mpl
# mpl.rc('figure',  figsize=(5, 5))
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
# VID = VID[:226, 300-226:,:]
ni, nj, nk = np.shape(VID)

invert = False

if invert:
    for i in range(nk):
        VID[:, :, i] = VID[:, :, i].max() - VID[:, :, i]
        

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


#%%
# ni, nj, nk = 700, 700, 700
# mi, mj, mk = 170, 170, 20


# VID_zn = np.random.random_integers(0, 255, (ni, nj, nk)).astype('float32')
# LUT_zn = np.random.random_integers(0, 255, (mi, mj, mk)).astype('float32')

#%% CuPy correlation
def corr_gpu(a, b):
    return a*cp.conj

cFT = lambda x: cp.fft.fftshift(cp.fft.fft2(x))
cIFT = lambda X: cp.fft.ifftshift(cp.fft.ifft2(X))
# cc = []
CC = np.empty((ni, nj, nk*mk), dtype='float16')
T0 = time.time()
T_CORR = []
# for i in tqdm(range(nk)):
for i in range(25):
    im = VID_zn[:, :, i]
    imft = cFT(cp.array(im))
    for j in range(mk):
        fm = cp.pad(cp.array(LUT_zn[:, :, j]), int((ni-mi)/2))
        fmft = cFT(fm)
        # cc.append(cp.abs(cIFT(imft*cp.conj(fmft))).get().astype('float16'))
        # CC[:, :, i*mk+j] = np.abs(cIFT(corr_gpu(imft, fmft)))
        CC[:, :, i*mk+j] = cp.abs(cIFT(imft*cp.conj(fmft))).get().astype('float16')
        T_CORR.append((time.time()-T0)/60)
print(T_CORR[-1])

# CC = np.float16(CC)
# np.save('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\NEW ANALYSIS\\CC_f16_binaryLUT.npy', CC)
# np.save('C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\NEW ANALYSIS\\CC_f16_LUT_every2VIDEO.npy', CC)

#%% Export images for theis
cc = CC[:,:,:25]
cc_p = np.max(cc, axis=2)

plt.imsave('z_project.png', cc_p)

plt.gca().set_axis_off()
# plt.imshow(cc_p, cmap='gray')
# plt.savefig('z_project_corr.png', dpi=300, bbox='tight')

# for i in range(25):
#     plt.gca().set_axis_off()
#     # plt.imshow(cc[:,:,i], cmap='gray')
#     plt.imsave('corr_'+str(i)+'.png', cc[:,:,i])


#%% Video settings
magnification = 40          # Archea: 20, E. coli: 40, 40, 40, MAY 20 (new 20), Colloids 20
frame_rate = 60              # Archea: 30/5, E. coli: 60, 60, 60, MAY 100, Colloids 50
fs = 0.711*(magnification/10)                  # px/um
ps = (1 / fs)                    # Pixel size in image /um
SZ = 5                     # step size of LUT [Archea: 10um,E. coli: 20, 40, 20, MAY 20 (new 10)], Colloids: 10
number_of_images = nk      # Archea = 400 , Ecoli = 430, 430, 700  # MAY 275(550)
number_of_filters = mk      # Archea =  25 ,   Ecoli =  19,  19,  20  # MAY 30 (new 40)  

#%% Read correlation images file names OPTICAL ONLY
from skimage import restoration
import easygui as gui
from natsort import natsorted
import os
from scipy.ndimage import sobel, gaussian_filter

path = gui.diropenbox()
path = path + '/'

file_list = os.listdir(path)

need_sort = True
if need_sort:
    file_list = natsorted(file_list)


number_of_images = 430 #700 #430 # MAY 275(550) # Archea 400     ### NEW Ar 400  Ec 430
number_of_filters = 25 #20 #19   # MAY 30 (new 40)  # Archea 25  ### NEW Ar 24   Ec 22

image_number = []
filter_number = []
for k in tqdm(range(len(file_list))):
    filter_number.append(int(file_list[k][:2]))
    image_number.append(int(file_list[k][4:9]))

image_number = np.array(image_number)
filter_number = np.array(filter_number)

#%% Order arrays according to image-filter combination (iterate filter first) OPTICAL ONLY
flist = []
number = number_of_images*number_of_filters
fnum = np.empty(number)
inum = np.empty(number)
filter_number = np.array(filter_number)
image_number = np.array(image_number)

for k in range(number):
    k_image_index = np.where(image_number == k)[0]
    
    for index in k_image_index:
        flist.append(file_list[index])
    
    fnum[k*number_of_filters:k*number_of_filters+number_of_filters] = filter_number[k_image_index]
    inum[k*number_of_filters:k*number_of_filters+number_of_filters] = image_number[k_image_index]
    
file_list = flist

#%%
import cv2
# tt = cv2.imread(path+file_list[0])
# tt = tt[:,:,0]
# nii, njj = tt.shape

CC = [cv2.imread(path+file, 0) for file in tqdm(file_list)]



#%% Analysis with MAx value (good results)
# CC = np.load(gui.fileopenbox())

window = 3                                          # Set by window in peak_gauss_fit_analysis() function
w = 1                                               # Windos for quadratic fit in Z
pol = lambda a, x: a[0]*x**2 + a[1]*x + a[2]
pos = []

nii, njj = CC[0].shape

num_images = nk*mk

methods = ['GPU', 'Optical']
method = methods[1]

apply_filters = True  # Just for Optical

for k in tqdm(range(nk)):
# for k in range(2):
    
    if method == 'Optical':
        temp = np.empty((nii, njj, mk))
        ids = np.arange(k*mk, k*mk+mk) 
        
        for i, id in enumerate(ids):
            # print(file_list[id])
            
            # t = cv2.imread(path+file_list[id], 0)
            # t = plt.imread(path+file_list[id])
            # t = t[:,:,0]
            # temp[:, :, i] = t
            
            temp[:, :, i] = CC[id]
            
            if apply_filters:
                temp[:, :, i] = gaussian_filter(np.abs(sobel(temp[:, :, i])), 1)
         
        zp = np.max(temp, axis=2)
        zp_gauss = gaussian_filter(zp.astype('float32'), sigma=3)
        # zp_gauss = zp
        
        
        # r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.5, min_distance=50, num_peaks=1)
        r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.8, min_distance=15)
        
        # plt.imshow(zp_gauss, cmap='gray')
        # plt.scatter(r[:, 1], r[:, 0])
    
    elif method == 'GPU':
        temp = CC[:, :, k*mk:k*mk+mk]
        zp = np.max(temp, axis=2)
        zp_gauss = gaussian_filter(zp.astype('float32'), sigma=3)
        # zp_gauss = zp
        
        # r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.2, min_distance=2, num_peaks=1)
        r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.3, min_distance=20)
        # print(r)
        # print('-')
        # print(r*ps)
        
        # plt.imshow(zp_gauss, cmap='gray')
        # plt.scatter(r[:, 1], r[:, 0])
    
    for r0 in r: 
        ri, rj = r0[0], r0[1]
        zpp = temp[ri-window:ri+window, rj-window:rj+window, :]
        
        # zpp_sum = np.sum(zpp, axis=(0, 1))
        zpp_sum = np.max(zpp, axis=(0,1))
        # plt.plot(zpp_sum, '.-')
        
        idmax = np.where(zpp_sum == zpp_sum.max())[0][0]
        # filter_sel = np.where(zpp_sum == zpp_sum.max())[0][0]
        
        if idmax > w and idmax < mk-w:
            ids = np.arange(idmax-w, idmax+w+1)
            ids_vals = zpp_sum[ids]
            coefs = np.polyfit(ids, np.float32(ids_vals), 2)
            
            interp_ids = np.linspace(ids[0], ids[-1], 100)
            interp_val = pol(coefs, interp_ids)
            
            # plt.plot(ids, ids_vals, 'H')
            # plt.plot(interp_ids, interp_val, '.-')
    
            filter_sel = interp_ids[interp_val == interp_val.max()][0] 
            # print(filter_sel)
        
        else:
            filter_sel = np.where(zpp_sum == zpp_sum.max())[0][0]
            # print(filter_sel)
        
        pos.append([ri, rj, filter_sel, k])

locs = np.array(pos)

#% Positions 3D Data Frame
posi = locs[:, 0]*ps
posj = locs[:, 1]*ps
post = locs[:, 3]/frame_rate
posframe = locs[:, 3]

true_z_of_target_im_1 = 121.1 # 96.1       #um
# zz = np.arange(number_of_filters-1, -1, -1)*SZ
zz = np.linspace(true_z_of_target_im_1, SZ, mk)
posk = np.empty_like(locs[:, 2])
for k in range(len(posk)):
    # posk[k] = zz[int(locs[k, 2])]
    posk[k] = true_z_of_target_im_1 - locs[k, 2]*SZ
       
data_3d = pd.DataFrame(np.transpose([posi.round(2), posj.round(2), posk.round(2), post.round(3), posframe]), columns=['X', 'Y', 'Z', 'TIME', 'FRAME'])


#%%
export = False

filename  = path_vid[:-4]+'_sig02_TH075_PMD20_OPT.csv'
if export:
    data_3d.to_csv(filename, index=False)


#%% Analysis with Gaussian Fitting (not good results)

window = 3  # Set by window in peak_gauss_fit_analysis() function
step = 3
pos = []

for k in tqdm(range(nk)):
    temp = CC[:, :, k*mk:k*mk+mk]
    zp = np.max(temp, axis=2)
    zp_gauss = gaussian_filter(zp.astype('float32'), sigma=1)
    # zp_gauss = zp
    
    r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.45, min_distance=20)
    
    # plt.imshow(zp_gauss, cmap='gray')
    # plt.scatter(r[:, 1], r[:, 0])
    
    for r0 in r: 
        ri, rj = r0[0], r0[1]
        zpp = temp[ri-window:ri+window, rj-window:rj+window, :]
        
        if k==0:
            std = []
            for i in range(zpp.shape[-1]):
                t = zpp[:, :, i]
                fit = f.peak_gauss_fit_analysis(np.float32(t))
                
                if fit == 'Empty':
                    std.append(np.nan)
                else:
                    std.append(fit[1])
                # std = np.array(std)
                
            if np.isnan(std).all():
                continue
            else:
                filter_sel = np.where(std == np.nanmin(std))[0][0]
            pos.append([ri, rj, filter_sel, k])
                
        else:
    #%
            center_index =int(pos[0][2])
            indices = np.arange(center_index-step, center_index+step+1)
            
            if (indices < 0).any():                                                     # Border handling
                indices_bool = indices < 0
                indices = indices[~indices_bool]
                
            elif (indices > number_of_filters-1).any():
                indices_bool = indices > number_of_filters-1
                indices = indices[~indices_bool]
            
            std = []
            for i in indices:
                t = zpp[:,  :, i]
                fit = f.peak_gauss_fit_analysis(np.float32(t))
                
                if fit == 'Empty':
                    std.append(np.nan)
                else:
                    std.append(fit[1])
            # std = np.array(std)
            
            if np.isnan(std).all():
                filter_sel = pos[-1][2]
            else:
                filter_sel = np.where(std == np.nanmin(std))[0][0]
    #%
            # zpp_sum = np.sum(zpp, axis=(0, 1))
            # zpp_sum = np.max(zpp, axis=(0,1))
            # filter_sel = np.where(zpp_sum == zpp_sum.max())[0][0]
            pos.append([ri, rj, filter_sel, k])

locs = np.array(pos)


#% Positions 3D Data Frame
posi = locs[:, 0]*ps
posj = locs[:, 1]*ps
post = locs[:, 3]/frame_rate
posframe = locs[:, 3]

true_z_of_target_im_1 = 120      #um
# zz = np.arange(number_of_filters-1, -1, -1)*SZ
zz = np.linspace(true_z_of_target_im_1, 5, 22)
posk = np.empty_like(locs[:, 2])
for k in range(len(posk)):
    posk[k] = zz[int(locs[k, 2])]
    
data_3d = pd.DataFrame(np.transpose([posj, posi, posk, post, posframe]), columns=['X', 'Y', 'Z', 'TIME', 'FRAME'])


#%% 3D 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = plt.figure(2)
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(data_3d['X'], data_3d['Y'], data_3d['Z'], c=data_3d['TIME'], marker='.')

pyplot.show()

#%%
frames = np.unique(data_3d['FRAME'])
for i in range(len(frames)):
# for i in range(1000):
    data = data_3d[data_3d['FRAME'] == i]
    plt.plot(data['Y'], -data['X'], '.')

#%% DBSCAN
import os
import sklearn.cluster as cl

cores = os.cpu_count()
eps = 20
min_samples = 20

# time.sleep(10)
T0_DBSCAN = time.time()
DBSCAN = cl.DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=cores).fit(data_3d[['X', 'Y', 'Z']])
LINKED = data_3d.copy()
LINKED['PARTICLE'] = DBSCAN.labels_
LINKED = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
LINKED['X'] = LINKED['X'] 

LINKED['Y'] = LINKED['Y']
T_DBSCAN = time.time() - T0_DBSCAN
print('T_DBSCAN', T_DBSCAN)

#%% Remove not swimming tracks

# particle_num = np.unique(LINKED['PARTICLE'])

# pp = []
# for p in particle_num:
#     L = LINKED[LINKED['PARTICLE'] == p]
#     # _, _, D = f.MSD(L.X.values, L.Y.values, L.Z.values, np.diff(L.TIME).mean())
    
#     xt = L['X'].max() - L['X'].min()
#     yt = L['Y'].max() - L['Y'].min()
#     zt = L['Z'].max() - L['Z'].min()
#     rt = np.sqrt(xt**2+yt**2+zt**2)
#     print(rt)
#     if rt < 15:
#         pp.append(p)
#         # b = np.where(LINKED['PARTICLE'].values == p)
#         LINKED = LINKED[LINKED['PARTICLE'] != p]

#%% MSD
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

particle_num = np.unique(LINKED['PARTICLE'])

LIN = []
for p in particle_num:
    
    L = LINKED[LINKED['PARTICLE'] ==  p]
    if len(L) > 50:
        # temp = f.clean_tracks(L)
        
        if len(set(L.FRAME)) != len(L):
            # temp = f.clean_tracks(L)
            temp = f.clean_tracks_search_sphere(L, 15)
            L = pd.DataFrame.transpose(pd.DataFrame(temp, ['X', 'Y', 'Z', 'TIME', 'FRAME','PARTICLE']))
            # L = L.drop_duplicates(subset='TIME', keep='last')
       
            # fig = plt.figure(1)
            # ax1 = fig.add_subplot(111, projection='3d') 
            # ax1.scatter(L['Y'], L['X'], L['Z'], c=L['TIME'])
            # pyplot.show()
           
       
        _, swim = f.MSD(L.X.values, L.Y.values, L.Z.values)
        print('swim? = '+str(swim))
         
        if swim == False:
            LINKED = LINKED[LINKED['PARTICLE'] != p]
            
        elif swim == True:
            LIN.append(L)
      
    else:
        LINKED = LINKED[LINKED['PARTICLE'] != p]
 
LINKED = pd.concat(LIN)  
LINKED = LINKED.reset_index(drop=True)     

   
#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

particle_num = np.unique(LINKED['PARTICLE'])
fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

# ax1.scatter(L['Y'], L['X'], L['Z'], c=L['PARTICLE'], marker='.')

for p in LINKED.PARTICLE.unique():
# p = 0
    LL = LINKED[LINKED['PARTICLE'] == p]    
    # ax1.plot(LL['Y'], LL['X'], LL['Z'])
    ax1.scatter(LL['X'], LL['Y'], LL['Z'])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    pyplot.show()

#%% Clean tracks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
p = 1
S = LINKED[LINKED['PARTICLE'] == p]
xx, yy, zz, tt, _, _ = f.clean_tracks(S)

# ax1.plot(S['Y'], S['X'], S['Z'])
ax1.plot(xx, yy, zz, '.-')
# ax1.scatter(S['Y'], S['X'], S['Z'], c=S['TIME'])
# ax1.scatter(yy, xx, zz, c=tt)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

pyplot.show()    
        
    

#%% CSAPS Smoothing
# import functions as f

tracks_3d = LINKED.copy()
# tracks_3d = LL
spline_degree = 3  # 3 for cubic spline
particle_num = np.unique(tracks_3d.PARTICLE)
T0_smooth = time.time()

dframes = []
for pn in particle_num:
    # Do not use this
    # L = LINKED[LINKED.PARTICLE == pn].values
    # X = f.smooth_curve(L, spline_degree=spline_degree, lim=20, sc=3000)
    
    L = tracks_3d[tracks_3d.PARTICLE == pn]
    # temp = f.clean_tracks(L)
    # L = pd.DataFrame.transpose(pd.DataFrame(temp, ['X', 'Y', 'Z', 'TIME', 'FRAME','PARTICLE']))
    
    
    if len(L) < 100:
        continue
    X = f.csaps_smoothing(L, smoothing_condition=0.995, filter_data=False, limit=15)
    
    if X != -1:
        X.append(pn*np.ones_like(X[1]))
        sc = pd.DataFrame(X, ['X', 'Y', 'Z', 'TIME', 'PARTICLE'])
        # smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], X[3], pn*np.ones_like(X[1])), axis=1))) 
        dframes.append(pd.DataFrame.transpose(sc))

        
smoothed_curves_df = pd.concat(dframes)
# smoothed_curves_df['X'] = smoothed_curves_df['X']
# smoothed_curves_df['Y'] = smoothed_curves_df['Y']

T_smooth = time.time() - T0_smooth

 #%% Crete Data Frame with Speed

xx, yy, zz, tt, pp, sp = -1, -1, -1, -1, -1, -1

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    speed, x, y, z, t = f.get_speed(s)
    xx = np.hstack((xx, x))
    yy = np.hstack((yy, y))
    zz = np.hstack((zz, z))
    tt = np.hstack((tt, t))
    pp = np.hstack((pp, pn*np.ones(len(t))))
    sp = np.hstack((sp, speed))
    

tracks_w_speed = pd.DataFrame(np.transpose([xx[1:], yy[1:], zz[1:], tt[1:], pp[1:], sp[1:]]), columns=['X', 'Y', 'Z', 'TIME', 'PARTICLE', 'SPEED'])

fig = plt.figure(1, dpi=150)
ax = fig.add_subplot(111, projection='3d')


# p = ax.scatter(tracks_w_speed['Y'], tracks_w_speed['X'], tracks_w_speed['Z'], c=tracks_w_speed['SPEED'], marker='.', s=20)
# cbar = plt.colorbar(p)
# cbar.set_label('Speed ($\mu ms^{-1}$)')

for pn in particle_num:
    # s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    s = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
    ax.plot(s['X'], s['Y'], s['Z'], linewidth=2)
    # ax.scatter(s['Y'], s['X'], s['Z'], c=s['SPEED'])
    
ax.axis('tight')
ax.set_title('$\it{Escherichia \ Coli}$', fontsize=40)  # $\it{Escherichia \ Coli}$
ax.set_xlabel('y ($\mu$m)', fontsize=20)
ax.set_ylabel('x ($\mu$m)', fontsize=20)
ax.set_zlabel('-z ($\mu$m)', fontsize=20)


plt.figure(2)
plt.hist(tracks_w_speed['SPEED'], 15)
mean_speed = tracks_w_speed['SPEED'].mean()
print(mean_speed)
plt.title('Speed: $\mu$ = ' + str(np.float16(mean_speed)) + ' $\mu m s^{-1}$', fontsize=40)
plt.xlabel('Speed ($\mu m s^{-1}$)', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

pyplot.show()

#%% 4th order derivative for speed
T = 1/frame_rate
s = smoothed_curves_df.copy()
dx, dy, dz = [], [], []
# v = []
for i in range(2, len(s['T'])-2):
    ddx = (8*(s['X'][i+1] - s['X'][i-1]) - (s['X'][i+2] - s['X'][i-2])) / (12*T)
    ddy = (8*(s['Y'][i+1] - s['Y'][i-1]) - (s['Y'][i+2] - s['Y'][i-2])) / (12*T)
    ddz = (8*(s['Z'][i+1] - s['Z'][i-1]) - (s['Z'][i+2] - s['Z'][i-2])) / (12*T)
    dx.append(np.abs(ddx))
    dy.append(np.abs(ddy))
    dz.append(np.abs(ddz))
    # v.append(np.sqrt(ddx**2+ddy**2+ddz**2))
    
dx = np.pad(np.array(dx), 2)
dy = np.pad(np.array(dy), 2)
dz = np.pad(np.array(dz), 2)    
# v = np.pad(np.array(v), 2)
v = np.sqrt(dx**2 + dy**2 + dz**2)


from csaps import csaps, CubicSmoothingSpline

L = LINKED[LINKED['PARTICLE'] == 4]
xn = L.X.values
yn = L.Y.values
zn = L.Z.values
tn = L['T'].values

data = [xn, yn, zn]
ti = np.linspace(tn[0], tn[-1], 1*len(tn))

smooth_data = csaps(tn, data, ti, smooth=0.99)
x_smooth = smooth_data[0, :]
y_smooth = smooth_data[1, :]
z_smooth = smooth_data[2, :]

s = pd.DataFrame([x_smooth, y_smooth, z_smooth, tn], ['X', 'Y', 'Z', 'FRAME']).T

#%% 3D Line Plot smooth
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt
plt.rcParams['figure.dpi'] = 150 
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    # p = ax.scatter(s.Y, s.X, s.Z, c=v, marker='.', s=5)
    # ax.scatter(LINKED.Y, LINKED.X, LINKED.Z, '.', s=1)
    ax.plot(s.Y, s.X, s.Z, 'r-', linewidth=0.5, markersize=2)
    
# fig.colorbar(p)
# cbar = plt.colorbar(p)
# cbar.set_label('Speed ($\mu ms^{-1}$)')

ax.axis('tight')
ax.set_xlabel('x ($\mu$m)')
ax.set_ylabel('y ($\mu$m)')
ax.set_zlabel('-z ($\mu$m)')

pyplot.show()




