# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:26:49 2021

@author: eers500
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import easygui as gui
from tqdm import tqdm
from functions import create_filter
from natsort import natsorted

path = gui.diropenbox()
write_path = gui.diropenbox()
file_list = os.listdir(path)

#%
file_list = natsorted(file_list)
#%%
shape = [1000, 1000]
img = []
filters = []
filters_8bit = []

for i, file in enumerate(tqdm(file_list)):
    im = plt.imread(path+'\\'+file)
    img.append(im[:, :, 0])
    # img.append(im)
    t = create_filter(img[i], shape)
    filters.append(t[0])
    filters_8bit.append(np.int16(t[1]))

#%%
binary = True
for i in tqdm(np.arange(len(filters))):
    if binary:
        plt.imsave(write_path+'\\'+str(i)+'.png', filters[i], cmap='gray')
    else:
        plt.imsave(write_path+'\\'+str(i)+'.png', filters_8bit[i], cmap='gray')
    
#%%


#%% Test correlation
FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))
img = plt.imread(path+'\\'+file_list[0])
img = img[:, :, 0]
IM = FT(img)

PAD = img
phase_sel = np.exp(1j**np.pi*PAD/255)
filts = FT(phase_sel)
f = -255*np.angle(filts)
ff = np.zeros_like(f)
ff[f >= 0] = 255

# f = f-f.min()
# ff = 255 * f/f.max()
filt = np.uint8(ff)

C = np.real(IM*np.conj(filt))
plt.figure(1)
plt.imshow(C, cmap='jet')
plt.show()

#%
from scipy import ndimage
from scipy import signal

CC = signal.correlate2d(img, img)
plt.figure(2)
plt.imshow(CC, cmap='jet')


#%
plt.figure(3)
CCC = ndimage.filters.correlate(img, img)
plt.imshow(CCC, cmap='jet')

#%%
FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))

img = np.random.randint(0, 2, (256, 256))
IM = FT(img)

PAD = img
phase_sel = np.exp(1j**np.pi*PAD/255)
filts = FT(phase_sel)
f = -255*np.angle(filts)
ff = np.zeros_like(f)
ff[f >= 0] = 255
filt = np.uint8(ff)

C = np.real(IM*np.conj(filt))
plt.figure(1)
plt.imshow(C, cmap='jet')
plt.show()

from scipy import ndimage
CCC = ndimage.filters.correlate(img, img)
plt.figure(2)
plt.imshow(CCC, cmap='jet')

























