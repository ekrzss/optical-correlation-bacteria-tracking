#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:50:28 2019

@author: erick
"""

import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f

#%%
# Import LUT form images
# LUT = [cv2.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = [mpimg.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = np.swapaxes(np.swapaxes(LUT, 0, 1), 1, 2)

#%%
# Import Video correlate
VID = f.videoImport("/home/erick/Documents/PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')

#%%
from scipy import ndimage

A = np.zeros((np.shape(VID)[0], np.shape(VID)[1]))
B = np.zeros((np.shape(LUT)[0], np.shape(LUT)[1]))

A[VID[:, :, 2] >= np.mean(VID[:, :, 2])] = 255
B[LUT[:, :, 2] >= np.mean(LUT[:, :, 2])] = 255

CORR = ndimage.filters.correlate(A, B, mode='wrap')

#%%
# plt.imshow(CORR, cmap='gray')
# plt.show()

#%%
# plt.subplot(1,2,1)
# plt.imshow(A, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(B, cmap='gray')

#%%
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

fig = go.Figure(data=[go.Surface(z=CORR)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.show()
plot(fig)