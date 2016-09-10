"""
Python + OpenCVで画像を表示する
http://docs.opencv.org/3.1.0/dc/d2e/tutorial_py_image_display.html

4.2.04.tif: http://sipi.usc.edu/database/database.php?volume=misc からダウンロードしたもの
"""

import numpy as np
import cv2
import os.path
lenna = "4.2.04.tiff"
if os.path.exists(lenna):
    img = cv2.imread(lenna)
    cv2.imshow("Lenna", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
