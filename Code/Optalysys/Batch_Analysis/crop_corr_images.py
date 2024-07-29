# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:41:40 2022

@author: eers500
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import easygui as gui
from tqdm import tqdm
import multiprocessing
import itertools
from multiprocessing import cpu_count

def write_png(inp):
    pread = inp[0]
    pexp = inp[1]
    file = inp[2]
    # s = inp[3]
    im = plt.imread(pread+file)
    # plt.imsave(pexp+'\\'+file[:-3]+'png', im[4:752, 233:980], cmap='gray') # Ecoli
    plt.imsave(pexp+'\\'+file[:-3]+'png', im[217:535, 447:763], cmap='gray')  # Archea

def select_roi(target):
    target = cv2.applyColorMap(target, cv2.COLORMAP_JET)
    cv2.namedWindow("image", flags= cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.imshow("image", target)
    showCrosshair = True
    fromCenter = False

    rect = cv2.selectROI("image", target, showCrosshair, fromCenter)

    (x, y, w, h) = rect

    # Crop image
    roi = target[y : y+h, x:x+w]
    
    return rect, roi 


if __name__ == '__main__':
    path = gui.diropenbox()
    file_list = os.listdir(path)
    pread = list(itertools.repeat(path+'\\', len(file_list)))
    
    # a = [4, 233]
    # b = [752, 980]
    
    # plt.figure(2)
    # plt.imshow(im[4:752, 233:980], cmap='jet')
    # plt.show()

    path_exp = os.path.split(path)[0]+'\\'+os.path.split(path)[-1]+'_cropped'
    os.mkdir(path_exp)
    
    # for file in tqdm(file_list):
        # im = plt.imread(path+'\\'+file)
    #     exp_path = path_exp+'\\'+file
    #     plt.imsave(exp_path[:-3]+'png', im[4:752, 233:980], cmap='gray')
    
    pexp = list(itertools.repeat(path_exp, len(file_list)))
    
    pool = multiprocessing.Pool(cpu_count()-1)
    # pool.map(write_png, zip(pread, pexp, file_list))
    # pool.close()
    
    # pool = Pool(cpu_count())
    results = []
#     T0 = time.time()

    # print('Processing File: '+ os.path.split(PATH[k])[-1])
    for _ in tqdm(pool.imap_unordered(write_png, zip(pread, pexp, file_list)), total=len(file_list)):
        results.append(_)
    pool.close()
    pool.join()
    