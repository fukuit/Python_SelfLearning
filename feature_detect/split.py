import cv2

img = cv2.imread('img/image02.png')
height, width, _ = img.shape
imageA = img[0:round(height/2), :]
imageB = img[round(height/2):, :]
cv2.imwrite('img/image02A.png', imageA)
cv2.imwrite('img/image02B.png', imageB)
