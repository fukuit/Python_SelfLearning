""" https://github.com/fukuit/Python_SelfLearning/blob/master/read_excel.py をPandasを使うように書き直す
"""

import os.path
import pandas as pd
import matplotlib.pyplot as plt

xlfile = "temperature.xlsx"
if os.path.exists(xlfile):
    # xlfileの読み込み
    xls = pd.ExcelFile(xlfile)
    sheet1 = xls.sheet_names[0]
    # DataFrameの作成
    df = pd.DataFrame(xls.parse(sheet1))

    # TokyoとKyotoの気温の平均を表示
    print(df.mean())

    # TokyoとKyotoの気温の温度分布の差をグラフで可視化する
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), dpi=100)
    df.plot(ax=axes[0])
    df.plot(ax=axes[1], kind='box')
    df.plot(ax=axes[2], kind='hist', alpha=0.5)
    plt.show()
