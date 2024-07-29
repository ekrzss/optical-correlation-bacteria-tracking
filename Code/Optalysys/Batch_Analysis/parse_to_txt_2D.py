# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:45:35 2021

@author: eers500
"""

import numpy as np

np.savetxt('input_image_number.txt', input_num, fmt='%i', delimiter=',')
np.savetxt('input_filter_number.txt', filter_num, fmt='%i', delimiter=',')

#%%
d = np.empty((754*16000, 944), dtype='uint8')

for k in range(16000):
    d[k*754:k*754+754, :] = CAMERA_PHOTO[:, :, k] 
    print(k)

#%%
np.savetxt('camera_photo.txt', d, fmt='%i', delimiter=',')