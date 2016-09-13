""" 全ての画像ファイルを大津メソッドで2値化する
元画像が、sample.tiffであれば、sample.bin.pngのように名前を変更して保存する
各2値化画像の、黒い部分の面積の割合を表示する
"""

import numpy as np
import cv2
import os

def convert_and_count( dir, ext ):
    for root,dirs,files in os.walk(dir):
        for f in files:
            fname, fext = os.path.splitext(f)
            if fext == ext:
                img = cv2.imread(os.path.join(root, f), 0)
                blur = cv2.GaussianBlur(img,(5,5),0)
                ret,imgb = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(root, fname + ".bin.png"), imgb)
                cnt = 0
                for val in imgb.flat:
                    if val == 0:
                        cnt+=1
                ratio = cnt / img.size
                msg = "%s:\t%.5f" % (f, ratio)
                print( msg )

if __name__ == "__main__":
    convert_and_count( "img", ".tiff" )
