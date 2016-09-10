"""
Python+OpenCVで画像を2値化する
http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html

4.2.04.tif: http://sipi.usc.edu/database/database.php?volume=misc からダウンロードしたもの
"""

import cv2
import numpy as np
import os.path
from matplotlib import pyplot as plt
lenna = "4.2.04.tiff"

if os.path.exists(lenna):
    img = cv2.imread(lenna, 0)
    mblur = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(mblur, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(mblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(mblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gblur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th5 = cv2.threshold(gblur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', "Otsu's Thresholding", "Otsu's Thresholding(Gaussian Blured)"]
    images = [img, th1, th2, th3, th4, th5]

    for i in range(0, 6):
        plt.subplot(3, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
