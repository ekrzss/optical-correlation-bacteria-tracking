#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:59:50 2019

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import functions as f
from skimage.feature import peak_local_max

#%%
camera_photo = scipy.io.loadmat('camera_photo.mat')
_,_,_,camera_photo = camera_photo.values()

input_image_number = scipy.io.loadmat('input_image_number.mat')
_,_,_,input_image_number=input_image_number.values()

filter_image_number = scipy.io.loadmat('filter_image_number.mat')
_,_,_,filter_image_number=filter_image_number.values()

#%%
PKS = peak_local_max(camera_photo[:,:,0], min_distance=3, threshold_abs=15)

plt.imshow(camera_photo[:, :, 0])
plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', facecolors='none', s=80, edgecolors='r')
plt.show()

#%%
#import plotly.graph_objects as go
#from plotly.offline import plot
#
#fig = go.Figure(data=[go.Surface(z=camera_photo[:, :, 1])])
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
#fig.show()
#plot(fig)