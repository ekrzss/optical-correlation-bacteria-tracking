#!/usr/bin/env python
# coding: utf-8

# In[46]:



import numpy as np
import cupy as cp
import matplotlib as mpl
mpl.rc('figure',  figsize=(5, 5))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')
import scipy.io

import pandas as pd
import functions as f
import easygui as gui
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import time
from tqdm import tqdm
from numba import vectorize, jit
import ipywidgets as widgets
from IPython.display import display


# In[10]:


path_video = gui.fileopenbox()
path_lut = gui.fileopenbox()


# In[30]:


vid = f.videoImport(path_video, 0)
lut = f.videoImport(path_lut, 0)
ni, nj, _ = vid.shape
mi, mj, mk = lut.shape


# In[20]:


print(vid.shape)
print(lut.shape)
mk

#%% 
zs = [123.875, 118.875, 113.875, 108.875, 103.875, 98.875, 93.875, 88.875, 83.875, 78.875, 73.875, 68.875, 63.90625, 58.90625, 53.90625,
      48.90625, 43.90625, 38.875, 33.875, 28.890625, 23.890625, 18.890625]

nframes = [[17, 18, 20], [111, 115, 118, 121], [139, 140, 141, 142], [170, 171, 172, 173, 174], [194, 195], [217, 218, 219, 220, 221], [226, 227],
         [253, 255], [256, 257, 258, 269, 270], [283, 299], [304], [320, 321], [326, 338], [360, 361, 362], [424, 425, 426, 427, 428],
         [432, 433, 434, 435], [446, 457], [362], [477, 478, 479], [489, 490, 491], [510, 511, 512, 513, 514, 515, 516], [526, 527, 528, 529, 530, 531, 532, 533, 534]] 

print(len(zs), len(nframes))


# In[49]:


frames_selection = [i[0] for i in nframes]
nk = len(frames_selection)
frames_selection


# In[53]:


frames = np.empty((ni, nj, len(frames_selection)))
for k in range(len(frames_selection)):
    frames[:, :, k] = vid[: ,:, frames_selection[k]]


# In[54]:


frames_zn = np.empty_like(frames)
for k in range(nk):
    A = frames[:,:,k]
    frames_zn[:, :, k] = (A-np.mean(A))/np.std(A)

lut_zn = np.empty_like(lut)
for k in range(mk):
    A = lut[:,:,k]
    lut_zn[:, :, k] = (A-np.mean(A))/np.std(A)


# In[55]:


def corr_gpu(a, b):
    return a*cp.conj

cFT = lambda x: cp.fft.fftshift(cp.fft.fft2(x))
cIFT = lambda X: cp.fft.ifftshift(cp.fft.ifft2(X))

CC = np.empty((ni, nj, nk*mk), dtype='float16')
T0 = time.time()
T_CORR = []
for i in tqdm(range(nk)):
# for i in range(10):
    im = frames_zn[:, :, i]
    imft = cFT(cp.array(im))
    for j in range(mk):
        fm = cp.pad(cp.array(lut_zn[:, :, j]), int((ni-mi)/2))
        fmft = cFT(fm)
        # CC[:, :, i*mk+j] = np.abs(cIFT(corr_gpu(imft, fmft)))
        CC[:, :, i*mk+j] = cp.abs(cIFT(imft*cp.conj(fmft))).get().astype('float16')
        T_CORR.append((time.time()-T0)/60)
print(T_CORR[-1])


# In[57]:


f.imshow_slider(CC, 2, 'gray')


# In[58]:


magnification = 20          # Archea: 20, E. coli: 40, 40, 40, MAY 20 (new 20), Colloids 20
frame_rate = 100              # Archea: 30/5, E. coli: 60, 60, 60, MAY 100, Colloids 50
fs = 0.711*(magnification/10)                  # px/um
ps = (1 / fs)                    # Pixel size in image /um
SZ = 5                     # step size of LUT [Archea: 10um,E. coli: 20, 40, 20, MAY 20 (new 10)], Colloids: 10
number_of_images = nk      # Archea = 400 , Ecoli = 430, 430, 700  # MAY 275(550)
number_of_filters = mk      # Archea =  25 ,   Ecoli =  19,  19,  20  # MAY 30 (new 40) 


# In[84]:


#%% Analysis with MAx value (good results)
# CC = np.load(gui.fileopenbox())

window = 3                                          # Set by window in peak_gauss_fit_analysis() function
w = 2                                               # Windos for quadratic fit in Z
pol = lambda a, x: a[0]*x**2 + a[1]*x + a[2]
pos = []

nii = 743
njj = 743
num_images = nk*mk

methods = ['GPU', 'Optical']
method = methods[0]

apply_filters = True

for k in tqdm(range(nk)):
# for k in range(2):
    
    if method == 'Optical':
        temp = np.empty((nii, njj, mk))
        ids = np.arange(k*mk, k*mk+mk) 
        
        for i, id in enumerate(ids):
            # print(file_list[id])
            t = plt.imread(path+file_list[id])        
            temp[:, :, i] = t[8:8+nii, 232:232+njj]
            
            if apply_filters:
                temp[:, :, i] = gaussian_filter(np.abs(sobel(temp[:, :, i])), 4)
         
        zp = np.max(temp, axis=2)
        zp_gauss = gaussian_filter(zp.astype('float32'), sigma=3)
        r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.6, min_distance=20)
        
        # plt.imshow(zp_gauss, cmap='gray')
        # plt.scatter(r[:, 1], r[:, 0])
    
    elif method == 'GPU':
        temp = CC[:, :, k*mk:k*mk+mk]
        zp = np.max(temp, axis=2)
        zp_gauss = gaussian_filter(zp.astype('float32'), sigma=3)
        # zp_gauss = zp
        
        # r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.2, min_distance=2, num_peaks=1)
        r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.3, min_distance=20)
        # print(r)
        # print('-')
        # print(r*ps)
        
        # plt.imshow(zp_gauss, cmap='gray')
        # plt.scatter(r[:, 1], r[:, 0])
    
    for r0 in r: 
        ri, rj = r0[0], r0[1]
        zpp = temp[ri-window:ri+window, rj-window:rj+window, :]
        
        # zpp_sum = np.sum(zpp, axis=(0, 1))
        zpp_sum = np.max(zpp, axis=(0,1))
        # plt.plot(zpp_sum, '.-')
        
        idmax = np.where(zpp_sum == zpp_sum.max())[0][0]
        
        if idmax > 3 and idmax < mk-3:
            ids = np.arange(idmax-w, idmax+w+1)
            ids_vals = zpp_sum[ids]
            coefs = np.polyfit(ids, np.float32(ids_vals), 2)
            
            interp_ids = np.linspace(ids[0], ids[-1], 20)
            interp_val = pol(coefs, interp_ids)
            
            # plt.plot(ids, ids_vals, 'H')
            # plt.plot(interp_ids, interp_val, '.-')
    
            filter_sel = interp_ids[interp_val == interp_val.max()][0] 
        
        else:
            filter_sel = np.where(zpp_sum == zpp_sum.max())[0][0]
        
        pos.append([ri, rj, filter_sel, k])

locs = np.array(pos)

#% Positions 3D Data Frame
posi = locs[:, 0]*ps
posj = locs[:, 1]*ps
post = locs[:, 3]/frame_rate
posframe = locs[:, 3]

true_z_of_target_im_1 = 121.1 # 96.1       #um
# zz = np.arange(number_of_filters-1, -1, -1)*SZ
zz = np.linspace(true_z_of_target_im_1, SZ, mk)
posk = np.empty_like(locs[:, 2])
for k in range(len(posk)):
    # posk[k] = zz[int(locs[k, 2])]
    posk[k] = true_z_of_target_im_1 - locs[k, 2]*SZ
    
    
data_3d = pd.DataFrame(np.transpose([posj, posi, posk, post, posframe]), columns=['X', 'Y', 'Z', 'TIME', 'FRAME'])


# In[85]:


#%% 3D 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = plt.figure(2)
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(data_3d['Y'], data_3d['X'], data_3d['Z'], c=data_3d['TIME'], marker='.')

pyplot.show()


# In[111]:


#%% DBSCAN
import os
import sklearn.cluster as cl

cores = os.cpu_count()
eps = 15
min_samples = 5

# time.sleep(10)
T0_DBSCAN = time.time()
DBSCAN = cl.DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=cores).fit(data_3d[['X', 'Y', 'Z']])
LINKED = data_3d.copy()
LINKED['PARTICLE'] = DBSCAN.labels_
LINKED = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
LINKED['X'] = LINKED['X'] 

LINKED['Y'] = LINKED['Y']
T_DBSCAN = time.time() - T0_DBSCAN
print('T_DBSCAN', T_DBSCAN)


# In[112]:



particle_num = np.unique(LINKED['PARTICLE'])
fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

# ax1.scatter(L['Y'], L['X'], L['Z'], c=L['PARTICLE'], marker='.')


p = 0
LL = LINKED[LINKED['PARTICLE'] == p]    
# ax1.plot(LL['Y'], LL['X'], LL['Z'])
ax1.scatter(LL['Y'], LL['X'], LL['Z'])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
pyplot.show()


# In[113]:


LL


# In[114]:


diff = np.abs(np.diff(np.array(LL.Z)))
plt.stem(diff)


# In[106]:


diff


# In[ ]:




