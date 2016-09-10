"""Excelのデータを読み込んで、列の平均値を表示し、t検定を行い、グラフを描く
元データは気象庁からダウンロードして加工したもの

"""

import xlrd
import os.path
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

xlfile = "temperature.xlsx"
if os.path.exists(xlfile):
    xls = xlrd.open_workbook(xlfile)
    sh1 = xls.sheet_by_index(0)
    nrows = sh1.nrows-1
    ncols = sh1.ncols
    data = np.zeros(ncols*nrows).reshape((nrows, ncols))
    date = []
    for r in range(1, nrows+1):
        for c in range(0, ncols):
            if c == 0:
                d = xlrd.xldate.xldate_as_datetime(sh1.cell(r, c).value, xls.datemode)
                date.append(d)
            else:
                data[r-1, c] = sh1.cell(r, c).value
    tokyo = data[:, 1]
    kyoto = data[:, 2]

    # 平均値の表示
    msg = "Tokyo(mean):\t%.2f\nKyoto(mean):\t%.2f" % (tokyo.mean(), kyoto.mean())
    print(msg)

    # t検定(welchの方法)を実施
    # t<0で、p-value < 0.05/2 であれば、「東京のほうが京都より気温が低くない(片側検定)」という帰無仮説を棄却できる
    t, p = stats.ttest_ind(tokyo, kyoto, equal_var=False)
    msg = "T-Value:\t%.5f\nP-Value:\t%.5f" % (t, p)
    print(msg)

    # 折れ線グラフで気温を表示
    plt.plot(date, tokyo, label="東京")
    plt.plot(date, kyoto, label="京都")
    plt.legend()
    plt.show()
