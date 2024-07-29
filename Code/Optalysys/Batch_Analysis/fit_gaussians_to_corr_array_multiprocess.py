# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:38:18 2021

@author: eers500
"""

import numpy as np
import pandas as pd
# import easygui as gui
import matplotlib.pyplot as plt
# import functions as f
import time
from skimage import restoration
from skimage.feature import peak_local_max
from multiprocessing import Pool, Process, freeze_support, set_start_method
from multiprocessing import cpu_count
# from tqdm import tqdm
import tqdm


def gauss(x, x0, y, y0, sigma, MAX):
            # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
            return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def peak_gauss_fit_analysis(input2darray):
    # k = peak_number  # Peak number
    # sel_size = 15
    # DATA = normalized_input[peak_array[k][0]-sel_size:peak_array[k][0]+sel_size, peak_array[k][1]-sel_size:peak_array[k][1]+sel_size]

    # sel_size = 20
    
    si, sj = input2darray.shape
    dr = 20
    sel_size = dr
    rmid = [int(si/2), int(sj/2)]
    
    # T0 = time.time()
    temp_input = input2darray[int(si/2)-dr:int(si/2)+dr, int(sj/2)-dr:int(sj/2)+dr]
    pkss = peak_local_max(temp_input, num_peaks=10, threshold_rel=0.8)   
    pks = (rmid+pkss)-dr
    # print(time.time()-T0)
    
    if len(pks) == 0:
        return 'Empty'
    
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
        # dist_to_pks = np.sqrt(np.sum((pks-rmid)**2, axis=1))
    
        if len(pks) > 1:
            pks = np.array([pks[0]])
    

    INTENSITY = input2darray[pks[0][0], pks[0][1]]
    DATA = input2darray[pks[0][0]-sel_size:pks[0][0]+sel_size+1, pks[0][1]-sel_size:pks[0][1]+sel_size+1]  # centered in pks
    
    if DATA.shape != (2*sel_size+1, 2*sel_size+1):
        return 'Empty'
    
    else:
        
        centeri, centerj = int(np.floor(DATA.shape[0]/2)), int(np.floor(DATA.shape[1]/2))
        center_value = DATA[centeri, centerj]
        I, J = np.meshgrid(np.arange(DATA.shape[0]), np.arange(DATA.shape[1]))
        sig = np.linspace(0.1, 40, 200)
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
        
        # pbar.update(1)
        # print('working...')
        
        return INTENSITY, SIGMA_OPT, OP, fitted_gaussian, DATA


#%%   

if __name__=='__main__':
    print('\nLoading array...')
    # CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\'+'CC.npy')   # Ecoli Sample 2   
    CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\'+'CC.npy')   # Ecoli Sample 1 - 03
    # CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\'+'CC.npy')   # Ecoli Sample 1 - 06
    # CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07\\'+'CC.npy')
    # CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\ZNCC\\GPU\\C_corr_zn.npy')   # Ecoli Sample 2 - particle 4
    # CC = np.load('C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\ZNCC\\GPU\\C_corr_zn.npy')
    print('\nDone!')
    
    number_of_images = 430   # Archea = 400, 400 , Ecoli = 275, 400, 430, 700 # MAY 275
    number_of_filters = 19  #Archea = 39, 25 , Ecoli = 30, 20,  19, 20        # MAY 30
    std_dev = np.empty((number_of_images*number_of_filters))
    max_val = np.empty_like(std_dev)
    fit = np.empty((number_of_images* number_of_filters), dtype='object')
    data = np.empty((number_of_images* number_of_filters), dtype='object')
    R2 = np.empty((number_of_images*number_of_filters))
    
    inputs = np.empty(number_of_images*number_of_filters, dtype='object')
    # inputs = np.empty(10, dtype='object')
    for i in range(len(inputs)):
        inputs[i] = CC[:, :, i]
        
    T0 = time.time()
    # pbar = tqdm.tqdm(total=8170/cpu_count())
    pool = Pool(cpu_count())
    # res = pool.map(peak_gauss_fit_analysis, inputs)    
    
    res = []
    for _ in tqdm.tqdm(pool.imap_unordered(peak_gauss_fit_analysis, inputs), total=int(number_of_images*number_of_filters)):
        res.append(_)
    
    pool.close()
    pool.join()
    T = time.time() - T0
    print('\n Elapsed time: ', T/60, ' min')
    
    for i in range(len(res)):
        temp = res[:][i]
        max_val[i] = temp[0]
        std_dev[i] = temp[1]
        fit[i] = temp[3]
        data[i] = temp[4]
        e = data[i] - fit[i]
        ym = np.mean(data[i])
        SSres = np.sum(e**2)
        SStot = np.sum((data[i] - ym)**2)
        R2[i] = 1 - SSres/SStot
        
        
    max_val = np.reshape(max_val, (number_of_images, number_of_filters))
    std_dev = np.reshape(std_dev, (number_of_images, number_of_filters))
    fit = np.reshape(fit, (number_of_images, number_of_filters))
    data = np.reshape(data, (number_of_images, number_of_filters))
    R2 = np.reshape(R2, (number_of_images, number_of_filters))

#%% Import 2D track and load correaltion array

# pnumber = 35
# path_track = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\'+'Spots in tracks statistics_35.csv'
# # path_track = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07\\Spots in tracks statistics_p0.csv'
# # path_track = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
# # path_track = 'C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\Spots in tracks statistics_p3.csv'
# # path_track = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\Spots in tracks statistics_4.csv'

# track_data = pd.read_csv(path_track)
# r0_track_df = track_data[track_data['TRACK_ID'] == pnumber]
# r0_track = r0_track_df[['POSITION_Y', 'POSITION_X']].values        
#%%
# method = 'std_dev' # std_dev or max_val

# if method == 'std_dev':
#     std_val = np.nanmin(std_dev, axis=1)
#     coord_i = r0_track[:, 0]
#     coord_j = r0_track[:, 1]
#     std_min = np.zeros_like(std_dev, dtype=bool)
    
#     for i in range(len(std_val)):
#         vv = std_dev[i, :] == std_val[i]
#         v = np.where(std_dev[i, :] == std_val[i])[0]
#         if np.count_nonzero(vv) == 1:
#             std_min[i, v[0]] = True
        
#         elif np.count_nonzero(vv) > 1:
#             std_min[i, v[0]] = True 
    
#     image_match, filter_match = np.where(std_min == True)
#     coord_i = coord_i[image_match]
#     coord_j = coord_j[image_match]
    
# elif method == 'max_val':
#     max_vals = np.nanmin(max_val, axis=1)
#     coord_i = r0_track[:, 0]
#     coord_j = r0_track[:, 1]
#     val_max = np.zeros_like(max_val, dtype=bool)
    
#     for i in range(len(max_vals)):
#         vv = max_val[i, :] == max_vals[i]
#         v = np.where(max_val[i, :] == max_vals[i])[0]
#         if np.count_nonzero(vv) == 1:
#             val_max[i, v[0]] = True
        
#         elif np.count_nonzero(vv) > 1:
#             val_max[i, v[0]] = True 
    
#     image_match, filter_match = np.where(val_max == True)
#     coord_i = coord_i[image_match]
#     coord_j = coord_j[image_match]

# #%% Filter filter_match selection to avoid suddent jumps
# from scipy import ndimage
# import functions as f

# num = np.arange(len(filter_match)-1)
# jump = np.abs(np.diff(filter_match)) 
# smooth_jump = ndimage.gaussian_filter1d(jump, 1, mode='mirror')  # window of size 5 is arbitrary

# limit = 200*np.mean(smooth_jump)    # factor 2 is arbitrary

# filter_sel = filter_match[:-1]
# boolean = (jump >= 0) & (smooth_jump < limit)
# filtered = filter_sel[boolean]

# coord_ii = coord_i[:-1]
# coord_jj = coord_j[:-1]

# coord_ii = coord_ii[boolean]
# coord_jj = coord_jj[boolean]
# frame = num[boolean]


# # L = np.stack((coord_j, coord_i, filter_match), axis=1)
# L = np.stack((coord_jj, coord_ii, filtered), axis=1)
# LL = pd.DataFrame(L, columns=['X', 'Y', 'Z'])

# [x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 0.999, False)

# #%% 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
# #%matplotlib qt

# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(coord_j, coord_i, filter_match, c=np.linspace(0, 1, len(filter_match)), label='Detected Positions')
# ax.scatter(coord_jj, coord_ii, filtered, c=frame, label='Detected Positions')
# ax.plot(x_smooth, y_smooth, z_smooth, c='red', label='Detected Positions')

# # ax.set_xlabel(r'X [$\mu$m]')
# # ax.set_ylabel('Y [$\mu$m]')
# # ax.set_zlabel('Z[$\mu$m]')
# ax.set_title('Optical Correlation', fontsize=20)
# pyplot.show()
               
        

        
        
        
        



