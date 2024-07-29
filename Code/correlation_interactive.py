#%%
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy import ndimage

plt.set_cmap('gray')
plt.close()

IM = mpimg.imread('R.png')
IM_BINARY = np.zeros_like(IM)
IM_BINARY[IM >= 0.49] = 255

#%%
import numpy as np
import matplotlib.pyplot as plt
import functions as f

x = np.linspace(-20, 20, 40)
y = x
X, Y = np.meshgrid(x, y)
rho = np.sqrt(X**2 + Y**2)

A = f.fraunhofer(rho, 10, 50)
a = A + np.abs(np.min(A))
aa = 255*(a/np.max(a))
AA = np.zeros_like(aa)
AA[a >= 127] = 255
# plt.imshow(A, cmap='gray')

#%%
S = cv2.selectROI('IM_BINARY', np.uint8(IM*255), False, False)
IFILT = IM_BINARY[int(S[1]):int(S[1] + S[3]), int(S[0]):int(S[0] + S[2])]

#%%
# Correlation in Fourier space
FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

SI = np.shape(IM_BINARY)
S = np.shape(IFILT)
DSY = SI[0] - S[0]
DSX = SI[1] - S[1]

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

# I_FT = FT(IM)
I_FT = FT(IM_BINARY)
IFILT_FT = FT(IPAD)

R = I_FT * np.conj(IFILT_FT)
r = np.abs(IFT(R))

ix2, iy2 = np.where(r == np.max(r))
idx2, idy2 = ix2[0], iy2[0]

#%%
CORR = ndimage.correlate(IM_BINARY, IFILT, mode='reflect')
# CORR = ndimage.correlate(IM, IFILT, mode='wrap')
ix, iy = np.where(CORR == np.max(CORR))
idx, idy = ix[0], iy[0]

#%%
# Pyplot plot
ax1 = plt.subplot(2, 2, 1);  plt.imshow(IM, cmap='gray'); plt.title('Hologram'); plt.scatter(idy, idx, s=150, facecolors='none', edgecolors='r')
ax2 = plt.subplot(2, 2, 2); plt.imshow(IFILT, cmap='gray'); plt.title('Mask')
ax3 = plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1);  plt.imshow(CORR, cmap='gray'); plt.title('CORR'); plt.scatter(idy, idx, s=150, facecolors='none', edgecolors='r')
ax4 = plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1); plt.imshow(r, cmap='gray'); plt.title('r'); plt.scatter(idy2, idx2, s=150, facecolors='none', edgecolors='r')
f.dataCursor2D()
plt.show()

#%%
# import numpy as np
# import matplotlib.pyplot as plt
# import functions as f
#
# x = np.linspace(-20, 20, 40)
# y = x
# X, Y = np.meshgrid(x, y)
# rho = np.sqrt(X**2 + Y**2)
#
# A = f.fraunhofer(rho, 10, 50)
#
# plt.imshow(A, cmap='gray')