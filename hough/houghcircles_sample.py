import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/embryos.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                           dp=1, minDist=20, param1=50, param2=30,
                           minRadius=10, maxRadius=100)
circles = np.uint16(np.around(circles))
for (x, y, r) in circles[0]:
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.xticks([]),plt.yticks([])
plt.show()
