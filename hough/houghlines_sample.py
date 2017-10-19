'''
 $Lastupdate: Sat Jul 29 13:08:59 2017 $
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://imagej.nih.gov/ij/images/cat.jpg
img = cv2.imread('img/cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 120)

lines = cv2.HoughLinesP(edges, rho=5, theta=np.pi/180,
                        threshold=5, minLineLength=5, maxLineGap=5)
if lines is not None and len(lines)>0:
    for (x1, y1, x2, y2) in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.subplot(1,2,1),plt.imshow(edges, 'gray'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.xticks([]),plt.yticks([])
plt.show()
