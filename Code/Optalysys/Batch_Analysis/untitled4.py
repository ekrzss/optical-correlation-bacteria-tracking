# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:03:17 2021

@author: eers500
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from tqdm import tqdm

path = gui.fileopenbox()
vid = f.videoImport(path, 0)

#%%
v = vid[:,:,0]
im, _ = np.real(f.bandpassFilter(v, 3, 20))
imm = np.zeros_like(im)

boolean = im > 0.3*im.max() 
imm[boolean] = 255

im_gauss = gaussian_filter(imm, 2)
peaks = peak_local_max(im_gauss, threshold_rel=0.4)


plt.subplot(1,3,1); plt.imshow(im, cmap='gray')
plt.subplot(1,3,2); plt.imshow(imm, cmap='gray')
plt.subplot(1,3,3); plt.imshow(im_gauss, cmap='gray'); plt.scatter(peaks[:, 1], peaks[:, 0], c='red', marker='o')


#%%

pdata = []
imf = np.empty_like(vid)
ds = 15


for i in tqdm(range(vid.shape[-1])):
    v = vid[:,:,i]
    im, _ = np.real(f.bandpassFilter(v, 2, 30))
    imm = np.zeros_like(im)
    boolean = im > 0.3*im.max() 
    imm[boolean] = 255
    im_gauss = gaussian_filter(imm, 2)
    imf[:, :, i] = im_gauss
    peaks = peak_local_max(im_gauss, threshold_rel=0.4, min_distance=20)
    for p in peaks:
        pdata.append([p[0], p[1], i])
        
    # plt.imshow(im_gauss, cmap='gray')
    # plt.scatter(peaks[:, 1], peaks[:, 0], c='red', marker='o')
    # plt.title(str(i))
    # plt.pause(0.2)
    
pdata = np.array(pdata)

#%%
frames = np.unique(pdata[:, 2])
r = [[72, 114], [169, 30], [200, 32], [107, 118]]
r0 = r[0]
epsilon = 20
track = []
track.append([r0[0], r0[1]])

for frame in frames:
    fdata = pdata[pdata[:, 2] == frame, :2]
    
    diff = np.sum((fdata-r0)**2, axis=1)
    
    isnear = diff <= epsilon
    if isnear.any():
        r0 = fdata[isnear][0]
        
        
        # diff_min = np.sum((fdata-r0)**2, axis=1).min()
        # diff_min_id = np.where(diff == diff.min())[0][0]
    
    track.append([r0[0], r0[1]])
track = np.array(track)
    
plt.scatter(pdata[:, 1], -pdata[:, 0], c='red', marker='.')
plt.scatter(track[:, 1], -track[:, 0])
plt.show()
    
    
        







    