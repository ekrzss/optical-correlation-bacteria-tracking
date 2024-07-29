# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:08:50 2021

@author: eers500
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import time
from PIL import Image
import cv2
from tqdm import tqdm
from skimage import restoration
import easygui as gui
from natsort import natsorted
import functions as f

path = gui.diropenbox()
path = path + '\\'

# path = 'F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\correlation_results\\'
# path = 'F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\correlation_results_p58\\'
# path = 'F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\correlation_results_4x\\'
# path = 'F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\correlation_results_p58_4x\\'
# path = 'F:\\PhD\\E_coli\\June2021\\14\\sample_2\\correlation_results_p4_4x\\
# path = 'F:\\PhD\\E_coli\\June2021\\14\\sample_2\\correlation_results_p4\\'

file_list = os.listdir(path)

need_sort = True
if need_sort:
    file_list = natsorted(file_list)


number_of_images = 430 #700 #430 # MAY 275(550) # Archea 400     ### NEW Ar 400  Ec 430
number_of_filters = 25 #20 #19   # MAY 30 (new 40)  # Archea 25  ### NEW Ar 24   Ec 22

image_number = []
filter_number = []
for k in tqdm(range(len(file_list))):
    # image_number.append(int(file_list[k][7:11]))
    # filter_number.append(int(file_list[k][:4]))
    # filter_number.append(int(file_list[k][5:7]))
    # image_number.append(int(file_list[k][11:14]))
    filter_number.append(int(file_list[k][:2]))
    image_number.append(int(file_list[k][4:9]))
    
    # filter_number.append(int(file_list[k][:4]))
    # image_number.append(int(file_list[k][7:11]))

image_number = np.array(image_number)
filter_number = np.array(filter_number)

#%% Order arrays according to image-filter combination
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
    
    # cam[:, :, number*k:number*k+number] = frames[:, :, i_images]
    # cam[:, :, number_of_images*k:number_of_images*k+number_of_images] = frames[:, :, i_images]

#%%
ni = 743 #796 #200 #786 # May 80     # Archea 60
nj = 743  #1080 #225 #1064  # May 80  # Archea 60

nk = number_of_images*number_of_filters

C = np.empty((ni, nj, nk), dtype='float32')
T0 = time.time()
for k in tqdm(range(len(file_list))):
# for k in range(500):
    temp = plt.imread(path+file_list[k])
    # temp = np.array(Image.open(path+file_list[k]))
    # temp = cv2.imread(path+file_list[k], 0)
    # temp = temp[364:424, 500:560]                       # For Archea
    # temp = temp[353:433, 490:570]                     # For June samples
    # temp = temp[164:590, 318:744]
    # temp = temp[230:751,258:994] 
    # temp = temp[0:757, 228:975]
    # temp = temp[354:434, 491:571]                       # For May samples
    temp = temp[8:8+ni, 232:232+nj]
    # temp2 = np.uint8(255 * temp / temp.max())
    C[:, :, k] = temp
    # print(k)
T = time.time() - T0
# CC = C
C = np.float16(C)
# cam = np.empty((ni, nj, nk))
# filter_num = np.empty(nk, dtype='int')   
# image_num = np.empty(nk, dtype='int')  
    
# for k in tqdm(range(number_of_images)):
#     index_image = image_number == k
#     filter_num[k*number_of_filters:k*number_of_filters+number_of_filters] = filter_number[index_image]
#     image_num[k*number_of_filters:k*number_of_filters+number_of_filters] = image_number[index_image]
    
# comb = np.transpose(np.vstack((input_num, filter_num)))

#np.save('F:\\PhD\\E_coli\\June2021\\14\\sample_2\\'+'C.npy', C)
# np.save('F:\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_09us_06\\'+'C.npy', C)
# np.save('C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_2\\31_aug_21\\'+'CC_4x.npy', CC)
# CC = C

#%% Sobel + Gaussian filtering
from scipy.ndimage import sobel, gaussian_filter
# im = C[:,:,29]
# sob = np.abs(sobel(im))

# filt = gaussian_filter(sob, 1.5)

# plt.subplot(1,2,1); plt.imshow(im)
# plt.subplot(1,2,2); plt.imshow(filt)

CCC = np.empty_like(CC)
for k in tqdm(range(CCC.shape[-1])):
    CCC[:, :, k] = gaussian_filter(np.abs(sobel(CC[:, :, k])), 2)

#%%
n = 0
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(CC[:,:,n])
ax[1].imshow(CCC[:,:,n])

f.surf(CC[:, :, n])
f.surf(CCC[:, :, n])

#%% Rollin ball filter
# CC = np.empty_like(C)

# for k in tqdm(range(np.shape(CC)[-1])):
#     im = restoration.rolling_ball(C[:, :, k], radius=5)
#     CC[:, :, k] = C[:, :, k] - im

# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
# ax[0].imshow(C[:,:,0])
# ax[1].imshow(CC[:,:,0])