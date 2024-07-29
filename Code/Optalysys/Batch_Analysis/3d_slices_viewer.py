# -*- coding: utf-8 -*-
from __future__ import print_function
#from functions import dataCursor2D
import numpy as np
import matplotlib.pyplot as plt

# from opt_correlation_analysis import CAMERA_PHOTOS


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        # self.ind = self.slices//2
        self.ind = 0

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
#        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


fig, ax = plt.subplots(1, 1)

#gr =grad
#gr[gr<1]=0

a = 2 
#X = CORR[:, :, 0+a:21+a]
X = C
# X = rs
tracker = IndexTracker(ax, X)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

