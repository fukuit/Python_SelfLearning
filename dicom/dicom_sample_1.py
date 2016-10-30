### 複数のDICOM画像ファイルを読み込んで、3次元配列に格納する

import os
import dicom
from matplotlib import pyplot as plt
%matplotlib inline

root_dir = './BRAINIX'
dcms = []
for d, s, fl in os.walk(root_dir):
    for fn in fl:
        if ".dcm" in fn.lower():
            dcms.append(os.path.join(d, fn))
ref_dicom = dicom.read_file(dcms[0])
d_array = np.zeros((ref_dicom.Rows, ref_dicom.Columns, len(dcms)) , dtype=ref_dicom.pixel_array.dtype)
for dcm in dcms:
    d = dicom.read_file(dcm)
    d_array[:, :, dcms.index(dcm)] = d.pixel_array

print(d_array.shape)
plt.subplot(1, 3, 1)
plt.imshow(d_array[127, :, :])
plt.subplot(1, 3, 2)
plt.imshow(d_array[:, 127, :])
plt.subplot(1, 3, 3)
plt.imshow(d_array[:, :, 49])
