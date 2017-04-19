import cv2
from matplotlib import pyplot as plt

# 画像の読み込み
img1 = cv2.imread('img/utsu1.png')
img2 = cv2.imread('img/utsu2.png')

# グレー変換
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT特徴量の抽出と画像化
sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(gray1)
kp2 = sift.detect(gray2)
img1_sift = cv2.drawKeypoints(gray1, kp1, None, flags=4)
img2_sift = cv2.drawKeypoints(gray2, kp2, None, flags=4)
# 保存
plt.imsave('img/utsu1_sift.png', img1_sift)
plt.imsave('img/utsu2_sift.png', img2_sift)

# AKAZE特徴量の抽出と画像化
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)
img1_akaze = cv2.drawKeypoints(gray1, kp1, None, flags=4)
img2_akaze = cv2.drawKeypoints(gray2, kp2, None, flags=4)
# 保存
plt.imsave('img/utsu1_akaze.png', img1_akaze)
plt.imsave('img/utsu2_akaze.png', img2_akaze)

# BFMatcherの定義と画像化
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# 保存
plt.imsave('img/utsu1-utsu2-match1.png', img3)

# BFMatcherの定義とKNNを使ったマッチングと画像化
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# 保存
plt.imsave('img/utsu1-utsu2-match2.png', img4)
