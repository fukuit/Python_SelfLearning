"""
Python+OpenCVで画像を2値化する閾値を求めて、閾値以下の値をカットオフする

4.2.04.tif: http://sipi.usc.edu/database/database.php?volume=misc からダウンロードしたもの
"""

import cv2
import os
from matplotlib import pyplot as plt
lenna = "4.2.04.tiff"

if os.path.exists(lenna):
    orig = cv2.imread(lenna, 0)
    img = cv2.medianBlur(orig, 5)
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_low = img < ret
    img[img_low] = 0

    plt.subplot(1, 3, 1), plt.imshow(orig, 'gray')
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(img, 'gray')
    plt.title('Threshold')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(th, 'gray')
    plt.title('Binarize')
    plt.xticks([]), plt.yticks([])
    plt.show()
