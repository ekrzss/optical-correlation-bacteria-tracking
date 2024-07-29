# %%
def rgb2gray(img):
    ## Convert rgb image to grayscale using Y' = 0.299R'+0.587G' + 0.114B'
    # Input:     img - RBG image
    # Output: img_gs - Grayscale image
    import numpy as np
    [ni, nj, nk] = img.shape
    img_gs = np.empty([ni, nj])
    for ii in range(0, ni):
        for jj in range(0, nj):
            img_gs[ii, jj] = 0.299 * img[ii, jj, 0] + 0.587 * img[ii, jj, 1] + 0.114 * img[ii, jj, 2]

    return img_gs


# %%
def square_image(img):
    ## Make image square by adding rows or columns of the mean value of the image np.mean(img)
    # Input: img - grayscale image
    # Output: imgs - square image
    #         axis - axis where data is added
    #            d - number of rows/columns added
    import numpy as np

    [ni, nj] = img.shape
    dn = ni - nj
    d = abs(dn)
    if dn < 0:
        M = np.flip(img[ni - abs(dn):ni, :], 0)
        imgs = np.concatenate((img, M), axis=0)
        axis = 'i'
    elif dn > 0:
        M = np.flip(img[:, nj - abs(dn):nj], 1)
        imgs = np.concatenate((img, M), axis=1)
        axis = 'j'
    elif dn == 0:
        imgs = img
        axis = 'square'
    return imgs, axis, d


# %%
def bandpassFilter(img, xs, xl):
    ## Bandpass filter
    # Input: img - Grayscale image array (2D)
    #        xl  - Large cutoff size (Pixels)
    #        xs  - Small cutoff size (Pixels)
    # Output: img_filt - filtered image
    import numpy as np

    # FFT the grayscale image
    imgfft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(imgfft)
    img_amp = abs(img_fft)
    del imgfft

    # Pre filter image information
    [ni, nj] = img_amp.shape
    MIS = ni

    # Create bandpass filter when BigAxis ==
    LCO = np.empty([ni, nj])
    SCO = np.empty([ni, nj])

    for ii in range(0, ni - 1):
        for jj in range(0, nj - 1):
            LCO[ii, jj] = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xl / MIS) ** 2)
            SCO[ii, jj] = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xs / MIS) ** 2)
    BP = SCO - LCO
    BPP = np.fft.ifftshift(BP)
    # Filter image
    filtered = BP * img_fft
    img_filt = np.fft.ifftshift(filtered)
    img_filt = np.fft.ifft2(img_filt)
    # img_filt = np.rot90(np.real(img_filt),2)

    return img_filt, BPP


# %%
def videoImport(video, N):
    ## Import video as stack of images in a 3D array
    #   Input:  video   - path to video file
    #               N   - frame number to import
    #   Output: imStack - 3D array of stacked images in 8-bit
    import cv2
    import numpy as np

    CAP = cv2.VideoCapture(video)
    NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('uint8'))
    # IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))

    I = 0
    SUCCESS = True

    if N == 0:
        IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('float32'))
        IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))

        while (I < NUM_FRAMES and SUCCESS):
            SUCCESS, IMG[I] = CAP.read()
            IM_STACK[I] = IMG[I, :, :, 1]
            I += 1
            # print(('VI', I))

    elif N > 0:
        IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('float32'))
        IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))
        STACK = IM_STACK

        while (I < NUM_FRAMES and SUCCESS):
            SUCCESS, IMG[I] = CAP.read()
            STACK[I] = IMG[I, :, :, 1]
            if I == N:
                IM_STACK = IMG[I, :, :, 1]
                FRAMENUM = I
                print(('VI', I))
            I += 1
    CAP.release()

    if N == 0:
        IM_STACK = np.swapaxes(np.swapaxes(IM_STACK, 0, 2), 0, 1)

    return IM_STACK


# %%
def exportAVI(filename, IM, NI, NJ, fps):
    ## Export 3D array to .AVI movie file
    #   Input:  IM - numpy 3D array
    #           NI - number of rows of array
    #           NJ - number of columns of array
    #          fps - frames per second of output file
    #   Output: .AVI file in working folder
    import os
    from cv2 import VideoWriter, VideoWriter_fourcc
    dir = os.getcwd()
    filenames = os.path.join(dir, filename)
    FOURCC = VideoWriter_fourcc(*'MJPG')
    VIDEO = VideoWriter(filenames, FOURCC, float(fps), (NJ, NI), 0)

    for i in range(IM.shape[2]):
        frame = IM[:, :, i]
        #    frame = np.random.randint(0, 255, (NI, NJ,3)).astype('uint8')
        VIDEO.write(frame)

    VIDEO.release()

    print(filename, 'exported successfully')
    return


# %%
def rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS):
    ## Rayleigh-Sommerfeld Back Propagator
    #   Inputs:          I - hologram (grayscale)
    #             I_MEDIAN - median image
    #                    Z - numpy array defining defocusing distances
    #   Output:        IMM - 3D array representing stack of images at different Z
    import math as m
    import numpy as np
    from functions import bandpassFilter

    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
    IN = I / I_MEDIAN
    #    IN = I - I_MEDIAN,
    #     IN[IN < 0] = 0

    # Bandpass Filter
    _, BP = bandpassFilter(IN, 2, 30)
    E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))

    # Patameters     #Set as input parameters
    # N = 1.3226               # Index of refraction
    LAMBDA = LAMBDA       # HeNe
    FS = FS               # Sampling Frequency px/um
    NI = np.shape(IN)[0]  # Number of rows
    NJ = np.shape(IN)[1]  # Nymber of columns
    # SZ = 10
    Z = SZ*np.arange(0, NUMSTEPS)
    # Z = (FS * (51 / 31)) * np.arange(0, NUMSTEPS)
    #    Z = SZ*np.arange(0, NUMSTEPS)
    K = 2 * m.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    P = np.empty_like(I_MEDIAN, dtype='complex64')
    for i in range(NI):
        for j in range(NJ):
            P[i, j] = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2 * ((i - NI / 2) ** 2 + (j - NJ / 2) ** 2)

    # P = np.conj(P)
    Q = np.sqrt(1 - P) - 1

    if all(Z > 0):
        Q = np.conj(Q)

    # R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
    IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
        IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R)))
    #        print(('RS', k))
    return IZ


# %%
def medianImage(VID, numFrames):
    ## Median Image
    #   Input:   VID - 3D numpy array of video file
    #   Output: MEAN - 2D pixel mean array
    import numpy as np

    def spaced_elements(array, numElems):
        out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
        return out

    N = np.shape(VID)[2]
    id = spaced_elements(np.arange(N), numFrames)

    print('MI')
    MEAN = np.median(VID[:, :, id], axis=2)

    return MEAN


# %%
def zGradientStack(IM):
    # Z-Gradient Stack
    #   Inputs:   I - hologram (grayscale)
    #            IM - median image
    #             Z - numpy array defining defocusing distances
    #   Output: CONV - 3D array representing stack of images at different Z
    import numpy as np
    from scipy import ndimage
    from functions import rayleighSommerfeldPropagator, exportAVI

    #    I = mpimg.imread('131118-1.png')
    #    I_MEDIAN = mpimg.imread('AVG_131118-2.png')
    #    Z = 0.02*np.arange(1, 151)
    #     IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)

    # %% Sobel-type kernel
    SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
    SZ1 = np.zeros_like(SZ0)
    SZ2 = -SZ0
    SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
    del SZ0, SZ1, SZ2

    # %% Convolution IM*SZ
    # IM = IM ** 2
    IMM = np.dstack((IM[:, :, 0][:, :, np.newaxis], IM, IM[:, :, -1][:, :, np.newaxis]))
    GS = ndimage.convolve(IMM, SZ, mode='mirror')
    GS = np.delete(GS, [0, np.shape(GS)[2] - 1], axis=2)
    del IMM

    #    exportAVI('gradientStack.avi',CONV, CONV.shape[0], CONV.shape[1], 24)
    #    exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
    return GS


# %%
def dataCursor1D():
    # Data Cursor in plots
    import mpldatacursor
    mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                             formatter='x = {i}\ny = {y:.06g}'.format)
    return 0


# %%
def dataCursor2D():
    # Data Cursor in 2D plots
    import mpldatacursor
    mpldatacursor.datacursor(display='multiple', hover=True, bbox=dict(alpha=1, fc='w'),
                             formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
    return 0


# %%
def dataCursor3D():
    # Data Cursor in 3D plots
    import mpldatacursor
    mpldatacursor.datacursor(hover=False, bbox=dict(alpha=1, fc='w'),
                             formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
    return 0


# %%
def histeq(im):
    ## Histogram equalization of a grayscale image
    import numpy as np
    from PIL import Image

    # get image histogram
    imhist, bins = np.histogram(im.flatten(), 256, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


# %%
def guiImport():
    # GUI for values
    import PySimpleGUI as sg

    layout = [
        [sg.Text('Select AVi File recording', size=(35, 1)), sg.In(), sg.FileBrowse()],
        [sg.Text('Select Median Image (optional)', size=(35, 1)), sg.In(), sg.FileBrowse()],
        [sg.Text('Frame number to use for calculations', size=(35, 1)), sg.InputText(default_text=1)],
        [sg.Text('Number of frames for median image', size=(35, 1)), sg.InputText(default_text=20)],
        [sg.Text('Refraction index of media (water = 1.3226)', size=(35, 1)), sg.InputText(default_text=1.3226)],
        [sg.Text('Wavelength in um (HeNe/0.642)', size=(35, 1)), sg.InputText(default_text=0.642)],
        [sg.Text('Sampling Frequency px/um (0.711) ', size=(35, 1)), sg.InputText(default_text=0.711)],
        [sg.Text('Step size (10)', size=(35, 1)), sg.InputText(default_text=10)],
        [sg.Text('Number os steps (150)', size=(35, 1)), sg.InputText(default_text=150)],
        [sg.Text('Gradient Stack Threshold (~0.1)', size=(35, 1)), sg.InputText(default_text=0.1)],
        [sg.Text('Magnification (10, 20, etc)', size=(35, 1)), sg.InputText(default_text=10)],
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('Hologramphy inputs', layout)
    event, values = window.Read()
    window.Close()

    return values


# %%
# Particles positions in 3D
def positions3D(GS, FRAME_NUM):
    import numpy as np
    from skimage.feature import peak_local_max

    # LOCS = np.zeros((1, 4))
    # for k in range(GS.shape[2]):
    #     PEAKS = peak_local_max(GS[:, :, k], indices=True)  # Check for peak radius
    #     ZZ = np.ones((PEAKS.shape[0], 1)) * k
    #     FRAME = np.ones((ZZ.shape[0], 1)) * FRAME_NUM
    #     PEAKS = np.hstack((PEAKS, ZZ, FRAME))
    #     LOCS = np.append(LOCS, PEAKS, axis=0)
    # LOCS = np.delete(LOCS, 0, 0)

    ZP = np.max(GS, axis=2)
    PKS = peak_local_max(ZP, min_distance=3)

    MAX = np.empty((len(PKS), 1))
    for i in range(len(PKS)):
        M = np.where(GS[PKS[i, 0], PKS[i, 1], :] == np.max(GS[PKS[i, 0], PKS[i, 1], :]))
        MAX[i, 0] = M[0][0]

    PKS = np.hstack((PKS, MAX))

    return PKS


# %%
def plot3D(LOCS, title, fig, ax):
    # 3D Scatter Plot
    # from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot

    # fig = pyplot.figure()
    # ax = Axes3D(fig)

    ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, marker='o')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize='20')
    ax.set_xlabel('x (pixels)', fontsize='18')
    ax.set_ylabel('y (pixels)', fontsize='18')
    ax.set_zlabel('z (slices)', fontsize='18')
    pyplot.show()

    return

#%%
def imshow_sequence(im, delay, run):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    while run:
        for i in range(np.shape(im)[2]):
            ax.cla()
            ax.imshow(im[:, :, i], cmap='gray')
            ax.set_title("frame {}".format(i))
            # Note that using time.sleep does *not* work here!
            plt.pause(delay)

    return

#%%
# Fraunhofer diffraction
def fraunhofer(rho, wsize, zdist):
    """Fraunhofer diffraction"""
    import numpy as np
    from scipy.special import j1
    lam = 0.642     # lambda
    # z = 5
    return j1(np.pi*wsize*rho/(lam*zdist))/(wsize*rho/(lam*zdist))
