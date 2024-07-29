# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:14:56 2021

@author: eers500
"""
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time
from skimage import restoration
from skimage.feature import peak_local_max
from tqdm import tqdm

# CC, _ = f.histeq(C)
# CC = C
# CC = np.empty_like(C)

# for k in tqdm(range(np.shape(CC)[-1])):
#     im = restoration.rolling_ball(C[:, :, k], radius=3)
#     CC[:, :, k] = C[:, :, k] - im
#     # print(k)

# np.save('F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\'+'CC.npy', CC)
# CC = C
# CC = np.load('F:\\PhD\\E_coli\\June2021\\14\\sample_2\\'+'CC.npy')   # Ecoli Sample 2
# CC = np.load('F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\'+'CC.npy')   # Ecoli Sample 1 - 03
# CC = np.load('F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\'+'CC.npy')   # Ecoli Sample 1 - 06
CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07\\'+'CC.npy')

#%%
# plt.subplot(1, 2, 1)
# plt.imshow(C[:, :, 0], cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(CC[:, :, 0], cmap='gray')

#%%
def gauss(x, x0, y, y0, sigma, MAX):
           # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
           return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

# def peak_gauss_fit_analysis(normalized_input, peak_number, peak_array, sel_size):
def peak_gauss_fit_analysis(input2darray):
    # k = peak_number  # Peak number
    # sel_size = 15
    # DATA = normalized_input[peak_array[k][0]-sel_size:peak_array[k][0]+sel_size, peak_array[k][1]-sel_size:peak_array[k][1]+sel_size]

    sel_size = 10
    
    si, sj = input2darray.shape
    dr = 20
    rmid = [int(si/2), int(sj/2)]
    
    # T0 = time.time()
    temp_input = input2darray[int(si/2)-dr:int(si/2)+dr, int(sj/2)-dr:int(sj/2)+dr]
    pkss = peak_local_max(temp_input, num_peaks=10)   
    pks = (rmid+pkss)-dr
    # print(time.time()-T0)
    
    # T0 = time.time()
    # pkss = peak_local_max(input2darray, num_peaks=10)
    # print(time.time()-T0)
    dist_to_rmid = np.sqrt(np.sum((pks-rmid)**2, axis=1))
    # pks_near_rmid = pks[dist_to_rmid <= dr, :]
    pks_near_rmid = pks[dist_to_rmid == dist_to_rmid.min(), :]
    
    # plt.imshow(input2darray)
    # plt.scatter(r1[:,1], r1[:,0], c='red')
    # plt.scatter(pks[:,1], pks[:,0], c='yellow')
    
    if len(pks_near_rmid) == 0:
        pks = np.array([np.array(rmid)])
    else: 
    
        intensity_pks_near_rmid = np.empty(len(pks_near_rmid))
        for i in range(len(pks_near_rmid)):
            intensity_pks_near_rmid[i] = input2darray[pks_near_rmid[i][0], pks_near_rmid[i][1]]
    
        pks = pks_near_rmid[intensity_pks_near_rmid == intensity_pks_near_rmid.max()]
        dist_to_pks = np.sqrt(np.sum((pks-rmid)**2, axis=1))
    
        if len(pks) > 1:
            pks = np.array([pks[0]])
    

    INTENSITY = input2darray[pks[0][0], pks[0][1]]
    DATA = input2darray[pks[0][0]-sel_size:pks[0][0]+sel_size+1, pks[0][1]-sel_size:pks[0][1]+sel_size+1]  # centered in pks
    
    if DATA.shape != (2*sel_size+1, 2*sel_size+1):
        return 'Empty'
    
    # if DATA.shape == (0, 0):
    # if np.all(DATA.shape) != True:
    #     print('Peak not in center')
    #     return 'Empty'
    else:
        pks = peak_local_max(DATA, num_peaks=1)
        
        # def gauss(x, x0, y, y0, sigma, MAX):
        #     # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
        #     return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))    
        
        I, J = np.meshgrid(np.arange(DATA.shape[0]), np.arange(DATA.shape[1]))
        sig = np.linspace(0.1, 40, 200)
        # MAX = np.arange(256)
        # MAX= np.linspace(DATA.min(), DATA.max(), 200)
        # chisq = np.empty((len(sig), len(MAX)))
        chisq = np.empty_like(sig)
        
        for ii in range(len(sig)):
            chisq[ii] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], DATA.max()))**2)/np.var(DATA)
            # for jj in range(len(MAX)):
                # chisq[ii, jj] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], MAX[jj]))**2)/np.var(DATA)
                    
        LOC_MIN = np.where(chisq == np.min(chisq))
        SIGMA_OPT = sig[LOC_MIN[0][0]]
        # MAX_OPT = MAX[LOC_MIN[1][0]]
        fitted_gaussian = gauss(I, pks[0][1], J, pks[0][0], SIGMA_OPT, INTENSITY) #ZZ
        OP = np.sum(DATA)
        
        # plt.figure(1)
        # plt.plot(sig, chisq, '.-')
        # plt.figure(2)
        # plt.subplot(1, 2, 1); plt.imshow(DATA)
        # plt.subplot(1, 2, 2); plt.imshow(fitted_gaussian)
        
        return INTENSITY, SIGMA_OPT, OP, fitted_gaussian, DATA




#%%
number_of_images = 275   # Archea = 400, 400 , Ecoli = 275, 400, 430, 700, # MAY 275
number_of_filters = 30  #Archea = 39, 25 , Ecoli = 30, 20,  19, 20         # MAY 30
std_dev = np.empty((number_of_images, number_of_filters))
max_val = np.empty_like(std_dev)
fit = np.empty((number_of_images, number_of_filters), dtype='object')
data = np.empty((number_of_images, number_of_filters), dtype='object')
R2 = np.empty((number_of_images, number_of_filters))

T = []
T0 = time.time()
for i in tqdm(range(number_of_images)):
# for i in tqdm(range(10)):
    for j in range(number_of_filters):
        # print(i, j)
        temp_corr = CC[:, :, i*number_of_filters+j]
        vals = peak_gauss_fit_analysis(temp_corr)
        
        if vals == 'Empty':
            std_dev[i, j] = np.nan
            max_val[i, j] = np.nan
            fit[i, j] = np.nan
            data[i, j] = np.nan
            R2[i, j] = np.nan
        else:
            std_dev[i, j] = vals[1]
            max_val[i, j] = vals[0]
            fit[i, j] = vals[3]
            data[i, j] = vals[4]
            e = data[i, j] - fit[i, j]
            ym = np.mean(data[i, j])
            SSres = np.sum(e**2)
            SStot = np.sum((data[i, j] - ym)**2)
            R2[i, j] = 1 - SSres/SStot
            
    T.append(time.time()-T0)
        
print(T[-1])

#%%
plt.subplot(1,2,1)
plt.plot(R2[100, :])

plt.subplot(1,2,2)
plt.plot(std_dev[100, :])

#%%
plt.figure(1)
plt.subplot(2,2,1);plt.imshow(data[254, 4]/1E16)        
plt.subplot(2,2,2);plt.imshow(fit[254, 4]/1E16)

plt.subplot(2,2,3);plt.imshow(data[254, 15]/1E16)        
plt.subplot(2,2,4);plt.imshow(fit[254, 15]/1E16)

# plt.figure(2)
# plt.plot(max_val[254, :]/1E16)
# plt.plot(std_dev[254, :])

#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(max_val[254, :]/1E16, 'b.-', label='Max value of fit')


ax2 = ax1.twinx()
ax2.plot(std_dev[254, :], 'r.-')
ax1.plot(np.nan, 'r', label='Standard deviation of fit')


ax1.set_ylabel('Max Value [a. u.]')
ax1.set_xlabel('LUT image number')

ax2.set_ylabel('Standard deviation')
ax1.legend(loc='upper center')
plt.show()   
#%%
# std_val = np.min(std_dev, axis=1) # Need to select min vals differente than -1
std_val = np.nanmin(std_dev, axis=1)
# std_val = np.nanmax(R2, axis=1)

# std_val_locs = np.where(std_val != 'nan')[0]  # index of rows of selected values different from -1 and -2
# coord_i = r0_track[std_val_locs, 0]
# coord_j = r0_track[std_val_locs, 1]

coord_i = r0_track[:, 0]
coord_j = r0_track[:, 1]

# std_dev = std_dev[std_val_locs]

#%
# std_min = std_dev.min(axis=1,keepdims=1) == std_dev  # mask of min values of std_val. Rows correspond to filter number
std_min = np.zeros_like(std_dev, dtype=bool)

for i in range(len(std_val)):
    vv = std_dev[i, :] == std_val[i]
    v = np.where(std_dev[i, :] == std_val[i])[0]
    if np.count_nonzero(vv) == 1:
        std_min[i, v[0]] = True
    
    elif np.count_nonzero(vv) > 1:
        std_min[i, v[0]] = True 


#%
# for i in range(len(std_min)):
#     if std_min[i, :].all():
#         std_min[i, :] = False
            
        
# for i in range(len(std_min)):
#     v = np.where(std_min[i, :] == True)[0]
#     if std_min[i, :].all():
#         std_min[i, :] = False
        
#     elif len(v) > 1:
#         std_min[i, v[1:]] = False    
    
image_match, filter_match = np.where(std_min == True)
coord_i = coord_i[image_match]
coord_j = coord_j[image_match]
#%% Filter filter_match selection to avoid suddent jumps
from scipy import ndimage

num = np.arange(len(filter_match)-1)
jump = np.abs(np.diff(filter_match)) 
smooth_jump = ndimage.gaussian_filter1d(jump, 1, mode='mirror')  # window of size 5 is arbitrary

# plt.figure(1)
# plt.plot(50+jump, '.-') 
# plt.plot(smooth_jump, '.-')

limit = 2*np.mean(smooth_jump)    # factor 2 is arbitrary
# limit=1

filter_sel = filter_match[:-1]
boolean = (jump >= 0) & (smooth_jump < limit)
filtered = filter_sel[boolean]

# plt.figure(2)
# plt.plot(np.arange(len(filter_match)), 50+filter_match, '.-')
# plt.plot(num[boolean], filter_sel[boolean], '.-')

coord_ii = coord_i[:-1]
coord_jj = coord_j[:-1]

coord_ii = coord_ii[boolean]
coord_jj = coord_jj[boolean]
frame = num[boolean]


#%
#% CSAPS Smoothing
import functions as f

# L = np.stack((coord_j, coord_i, filter_match), axis=1)
L = np.stack((coord_jj, coord_ii, filtered), axis=1)
LL = pd.DataFrame(L, columns=['X', 'Y', 'Z'])

[x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 0.9999, False)
# [x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 1, True)

#% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt
#%

# zz = 20*np.arange(0, 20)
# z_pos = np.empty_like(filtered)
# for i in range(len(z_pos)):
#     z_pos[i] = zz[filtered[i]]

xx_shift = x_smooth-x_smooth.min()
xx_smooth = 100*xx_shift/xx_shift.max()

yy_shift = y_smooth-y_smooth.min()
yy_smooth = 100*yy_shift/yy_shift.max()

zz_shift = z_smooth
zz_smooth = 1-zz_shift/zz_shift.max()

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coord_j, coord_i, filter_match, c=np.linspace(0, 1, len(filter_match)), label='Detected Positions')
# ax.scatter(coord_jj, coord_ii, filtered, c=frame, label='Detected Positions')
# ax.scatter(coord_jj, coord_ii, z_pos, c=frame, label='Detected Positions')
# bb = y_smooth > 300
# ax.scatter(coord_jj[bb], coord_ii[bb], filtered[bb], c=frame[bb], label='Detected Positions')
# ax.plot(x_smooth[bb], y_smooth[bb], 5*(z_smooth.max()-z_smooth[bb])+4, c='red', label='Detected Positions')
ax.plot(x_smooth, y_smooth, z_smooth, c='red', label='Detected Positions')
# ax.plot(xx_smooth, yy_smooth, 100*zz_smooth, c='red', label='Detected Positions')
#ax.plot(xx_shift, yy_shift, 100*zz_smooth, c='blue', label='Detected Positions')

# ax.scatter(145, 45, 55, c='r')
# ax.scatter(135, 70, 38, c='r')
# ax.scatter(83, 181, 57, c='r')
# ax.scatter(49, 232, 62, c='r')
# ax.scatter(60, 213, 49, c='r')
# ax.scatter(23, 245, 45, c='r')
# ax.scatter(7, 289, 65, c='r')
# ax.scatter(0.4, 334, 68, c='r')
# ax.scatter(0.8, 311, 59, c='r')

# fig.gca().set_xlabel(r'wavelength $5000 \AA$')

# ax.plot(pos_i, pos_j, MAX_FILT, 'r-', label='Smoothed Curve')
ax.set_xlabel(r'X [$\mu$m]')
ax.set_ylabel('Y [$\mu$m]')
ax.set_zlabel('Z[$\mu$m]')
ax.set_title('Optical Correlation', fontsize=20)
pyplot.show()
        
#%%
# path = 'F:\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07\\Loc_track'


# f.exportAVI(path+'\loc_vid.avi', vid_surr, 80, 80, 30)






