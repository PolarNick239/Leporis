__author__ = "Polyarnyi Nickolay"

import cv2

from matplotlib import pyplot as plt
import matplotlib as mpl


def _imshow_bgr(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()


def _imshow_gray(img):
    plt.imshow(cmap=mpl.cm.Greys_r)
    plt.show()


def imshow(img):
    if len(img.shape) == 2:
        _imshow_gray(img)
    else:
        _imshow_bgr(img)
