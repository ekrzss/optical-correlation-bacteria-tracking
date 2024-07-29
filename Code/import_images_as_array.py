#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:15:46 2019

@author: erick
"""

import glob
#import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#%%
#images = [cv2.imread(file) for file in glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png")]

#%%

images = [mpimg.imread(file) for file in np.sort(glob.glob("//media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/substack_CES/IMAGES/*.png"))]
images = np.swapaxes(np.swapaxes(images, 0, 1), 1, 2)

#%%
plt.imshow(images[:, :, 0], cmap='gray')
plt.show()