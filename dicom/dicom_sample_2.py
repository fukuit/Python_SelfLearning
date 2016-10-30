### 複数のDICOM画像ファイルを読み込んで、中央部を円形に切り抜く

import os
import dicom
from matplotlib import pyplot as plt
%matplotlib inline

# BRAINIXフォルダ以下のdcmファイルを読み込む
root_dir = './BRAINIX'
dcms = []
for d, s, fl in os.walk(root_dir):
    for fn in fl:
        if ".dcm" in fn.lower():
            dcms.append(os.path.join(d, fn))
ref_dicom = dicom.read_file(dcms[0])

# 円に切り抜くためのマスクを作成する
x, y = np.indices((ref_dicom.Rows, ref_dicom.Columns))
circle = (x - (ref_dicom.Columns / 2))**2 + (y - (ref_dicom.Rows / 2))**2 < 64**2
mask = circle.astype(int)

# 全ての画像にmask処理を実施する
d_array = np.zeros((ref_dicom.Rows, ref_dicom.Columns, len(dcms)), dtype=ref_dicom.pixel_array.dtype)
for dcm in dcms:
    d = dicom.read_file(dcm)
    img = d.pixel_array * mask
    d_array[:, :, dcms.index(dcm)] = img

# 3断面表示する
plt.subplot(1, 3, 1)
plt.imshow(d_array[127, :, :])
plt.subplot(1, 3, 2)
plt.imshow(d_array[:, 127, :])
plt.subplot(1, 3, 3)
plt.imshow(d_array[:, :, 49])
