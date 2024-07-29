#%%
# import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy import ndimage
from mpldatacursor import datacursor

plt.set_cmap('gray')
plt.close()

#%%
# INPUT = mpimg.imread('R_Binary.png')
IN = 1 - mpimg.imread('R.png')
IFILT = 1 - mpimg.imread('R_ring_Binary.png')
IFILT = IFILT[:, :, 0]
I = IN

#%%
BINARY = True
if BINARY == True:
    I[I < 0.49] = 0
    I[I >= 0.49] = 1
    # IFILT[IFILT < np.mean(I)] = 0
    # IFILT[IFILT >= np.mean(I)] = 255

#%%
CORR = ndimage.correlate(I, IFILT, mode='reflect')
ix, iy = np.where(CORR == np.max(CORR))
idx, idy = ix[0], iy[0]
#%%
# Correlation in Fourier space
FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

SI = np.shape(I)
S = np.shape(IFILT)
DSY = SI[0]-S[0]
DSX = SI[1]-S[1]

if DSY % 2 == 0 and DSX % 2 == 0:
    NY = int(DSY / 2)
    NX = int(DSX / 2)
    IPAD = np.pad(IFILT, ((NY, NY), (NX, NX)), 'constant', constant_values=0)
elif DSY % 2 == 1 and DSX % 2 == 1:
    NY = int(np.floor(DSY / 2))
    NX = int(np.floor(DSX / 2))
    IPAD = np.pad(IFILT, ((NY, NY + 1), (NX, NX + 1)), 'constant', constant_values=0)
elif DSY % 2 == 0 and DSX % 2 == 1:
    NY = int(DSY / 2)
    NX = int(np.floor(DSX / 2))
    IPAD = np.pad(IFILT, ((NY, NY), (NX, NX + 1)), 'constant', constant_values=0)
elif DSY % 2 == 1 and DSX % 2 == 0:
    NY = int(np.floor(DSY / 2))
    NX = int(DSX / 2)
    IPAD = np.pad(IFILT, ((NY, NY + 1), (NX, NX)), 'constant', constant_values=0)

I_FT = FT(I)
IFILT_FT = FT(IPAD)

R = I_FT*np.conj(IFILT_FT)
r = np.abs(IFT(R))

ix2, iy2 = np.where(r == np.max(r))
idx2, idy2 = ix2[0], iy2[0]


#%%
# Pyplot plot
plt.figure(1)
ax1 = plt.subplot(2, 2, 1);  plt.imshow(IN, cmap='gray'); plt.title('Hologram'); plt.scatter(idy, idx, s=150, facecolors='none', edgecolors='r')
ax2 = plt.subplot(2, 2, 2); plt.imshow(IFILT, cmap='gray'); plt.title('Mask')
ax3 = plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1);  plt.imshow(CORR, cmap='gray'); plt.title('CORR'); plt.scatter(idy, idx, s=150, facecolors='none', edgecolors='r')
ax4 = plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1); plt.imshow(r, cmap='gray'); plt.title('r'); plt.scatter(idy2, idx2, s=150, facecolors='none', edgecolors='r')
# f.dataCursor2D()
datacursor(hover=True)
plt.show()

#%%
# # 3D surace Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
# xi, yi = np.where(CORR == np.max(CORR))
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# X, Y = np.meshgrid(np.arange(1, 1025, 1), np.arange(1, 1025, 1))
# ax.plot_surface(X, Y, r)
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
#
# # MAX = np.mean(CORR)*np.ones_like(X)
# # MAX[xi[0], yi[0]] = np.max(CORR)
# # ax.plot_surface(X, Y, MAX)
# # f.dataCursor3D()
# datacursor(hover=True)
# pyplot.show()
