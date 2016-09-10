""" K-th Nearest Neighbourによる手書き文字認識のデモ
http://docs.opencv.org/3.1.0/d8/d4b/tutorial_py_knn_opencv.html
チュートリアルのままではエラーになるので修正
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

datafile = 'digits.png'
img = cv2.imread(datafile)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells)
train = x[:, :50].reshape(-1, 400).astype(np.float32)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# tutorialのままではエラーになる3行
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, results, neighbours, dist = knn.findNearest(test, 5)

matches = results == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/results.size
print(accuracy)
