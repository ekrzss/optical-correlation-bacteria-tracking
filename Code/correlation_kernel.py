import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mpldatacursor
from progress.bar import Bar

##
I = mpimg.imread('NORM_MF1_30Hz_200us_awaysection.png')
K = mpimg.imread('SampleRing.png')

IFT = np.fft.fftshift(np.fft.fft2(I))
KFT = np.fft.fftshift(np.fft.fft2(K, s=np.shape(IFT)))

R = IFT*np.conj(KFT)
CORR = np.real(np.fft.ifft2(np.fft.ifftshift(R)))
C = CORR
C[C < np.max(C)] = 0

##
plt.imshow(np.real(KFT), cmap='gray')
