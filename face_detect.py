""" 画像から顔を抽出する
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

4.2.04.tif: http://sipi.usc.edu/database/database.php?volume=misc からダウンロードしたもの
haarcascade_frontalface_default.xml, haarcascade_eye.xml: OpenCVの配布物に含まれる
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def facedetect(file):
    """ haarを使って顔を識別し、顔部分を矩形で囲んで表示する
    Args:
        file : 対象とする画像ファイル名
    """
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    lenna = "4.2.04.tiff"
    if os.path.exists(lenna):
        facedetect(lenna)
