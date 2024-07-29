# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:03:17 2021

@author: eers500
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui
from scipy.ndimage import gaussian_filter, sobel, maximum_filter, minimum_filter
from skimage.feature import peak_local_max
from tqdm import tqdm
from skimage import restoration

path = gui.fileopenbox()
vid = f.videoImport(path, 0)

#%%
eps_gauss = 0.7
threshold_pcg = 0.12

# i=200

pks = -np.ones((1, 3))
for i in tqdm(range(number_of_images)):
    temp = CC[:, :, i*number_of_filters:i*number_of_filters+number_of_filters]
    imz = np.max(temp, axis=2)
    p = peak_local_max(imz, threshold_rel=0.3, min_distance=20)
    pp = i*np.ones((len(p), 1))
    pk = np.hstack((p, pp))
    pks = np.vstack((pks, pk))

pks = pks[1:, :]

# plt.imshow(imz)
# plt.scatter(pks[:, 1], pks[:, 0], facecolors='none', edgecolors='r')
plt.scatter(pks[:,1], -pks[:,0], marker='.')
plt.show()

#%%

for k in range(number_of_fiters):
    im, _ = np.real(f.bandpassFilter(VID[:, :, number_of_fiters], 2, 30))
    imm = np.zeros_like(im) 
    imm[im > threshold_pcg*im.max()] = 255
    im_gauss = gaussian_filter(imm, eps_gauss)
    
    r = peak_local_max(im_gauss, threshold_rel=0.7, min_distance=10)

    