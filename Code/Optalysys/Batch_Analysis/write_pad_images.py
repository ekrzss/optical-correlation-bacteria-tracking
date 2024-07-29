# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:37:01 2021

@author: eers500
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import easygui as gui
from tqdm import tqdm
from functions import create_filter

path = gui.diropenbox()
write_path = gui.diropenbox()
file_list = os.listdir(path)


#%%
ds = 300
for i, file in enumerate(tqdm(file_list)):
    im = plt.imread(path+'\\'+file)[:, :, 0]
    im_pad = np.pad(im, (300))
    plt.imsave(write_path+'\\'+file, im_pad, cmap='gray')



