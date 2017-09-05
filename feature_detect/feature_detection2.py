import cv2
import numpy as np
from matplotlib import pyplot as plt

def matchAB(fileA, fileB):
    # 画像の読み込み
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)

    # グレー変換
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # 画像サイズの取得
    height, width = grayA.shape
    # 部分画像を作って、マッチングさせる
    result_window = np.zeros((height, width), dtype=imgA.dtype)
    for start_y in range(0, height-100, 50):
        for start_x in range(0, width-100, 50):
            window = grayA[start_y:start_y+100, start_x:start_x+100]
            match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1]+100, max_loc[0]:max_loc[0]+100]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+100, start_x:start_x+100] = result

    # マッチングした結果できた差分画像の輪郭を抽出し、四角で囲む
    _, result_window_bin = cv2.threshold(result_window, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgC = imgA.copy()
    for contour in contours:
        min = np.nanmin(contour, 0)
        max = np.nanmax(contour, 0)
        loc1 = (min[0][0], min[0][1])
        loc2 = (max[0][0], max[0][1])
        cv2.rectangle(imgC, loc1, loc2, 255, 2)

    # 画像表示する
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)), plt.title('A'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)), plt.title('B'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.title('Answer'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    matchAB('img/image02A.png','img/image02B.png')
