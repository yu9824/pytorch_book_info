# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="5IMxZQV61P12"
# ## 講座3 Matplotlinb入門

# %% id="CVHsV3PP1P13"
# 日本語化ライブラリ導入
# !pip install japanize-matplotlib | tail -n 1

# %% id="pje_TMrPZIJQ"
# 共通事前処理のうち、Matplotlibに関係あるもの
import matplotlib.pyplot as plt

# matplotlib日本語化対応
import japanize_matplotlib

# %% id="gCLigwq3ZIJR"
# デフォルトパラメータ設定

# デフォルトフォントサイズ変更
# 都度設定する場合は、plt.legend(fontsize=14) など
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
# 都度設定する場合は plt.figure(figsize=(6,6))
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
# 都度設定する場合は plt.grid()
plt.rcParams['axes.grid'] = True

# %% id="EeFtwc6P1P18"
# 共通事前処理

# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

# 必要ライブラリのimport
import numpy as np

# %% [markdown] id="u-fIeytl1P1_"
# ### 1. pltによる簡易描画

# %% [markdown] id="udaKCL-L1P1_"
# #### 散布図

# %% id="Es-WNkfv1P1_"
# データ準備
import seaborn as sns
df_iris = sns.load_dataset("iris") 

# 結果確認
print(df_iris.head())

# 散布図x座標用配列
xs = df_iris['sepal_length'].values

# 散布図y座標用配列
ys = df_iris['sepal_width'].values

# %% id="UGxEsn0X1P2C"
# 散布図
plt.scatter(xs, ys)

# 描画
plt.show()


# %% [markdown] id="y7Ahnm431P2F"
# #### 関数グラフ

# %% id="1xx8yyay1P2F"
# データ準備

# シグモイド関数の定義
def sigmoid(x, a):
    return 1/(1 + np.exp(-a*x))

# グラフ描画用x座標リスト
xp = np.linspace(-3, 3, 61)
yp = sigmoid(xp, 1.0)
yp2 = sigmoid(xp, 2.0)

# %% [markdown] id="nARycg6a1P2K"
# #### 単純な例

# %% id="ProeQEHq1P2K"
# グラフ描画
plt.plot(xp, yp)

# 描画
plt.show()

# %% [markdown] id="Ksn0T5h71P2O"
# #### 複雑な例

# %% id="we62RKgU1P2O"
#ラベル付きグラフ描画 #1
plt.plot(xp, yp, 
         label='シグモイド関数1', lw=3, c='k')

# ラベル付きグラフ描画 #2
plt.plot(xp, yp2, 
         label='シグモイド関数2', lw=2, c='b')

# 凡例表示
plt.legend()

# 軸表示
plt.xlabel('x軸')
plt.ylabel('y軸')

# 描画
plt.show()

# %% [markdown] id="pXY-prC71P2W"
# ### 2. subplotを使った複数グラフの同時描画

# %% id="jEPCWTgQ1P2X"
# データ準備

# 手書き数字データ
# 時間がかかるので注意してください
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,as_frame=False)

# イメージデータ
image = mnist.data
# 正解データ
label = mnist.target

# %% id="n8IJVncG1P2Z"
# サイズ指定
plt.figure(figsize=(10, 3))

# 20個のイメージを表示
for i in range(20): 
    
    # i 番目のax変数取得
    ax = plt.subplot(2, 10, i+1)
    
    # i番目のイメージデータ取得し28x28に変換
    img = image[i].reshape(28,28)
    
    # imgをイメージ表示
    ax.imshow(img, cmap='gray_r')
    
    # 正解データをタイトル表示
    ax.set_title(label[i])
    
    # x, y目盛非表示
    ax.set_xticks([])
    ax.set_yticks([])
    
# 隣接オブジェクトとぶつからないようにする
plt.tight_layout()

# 表示
plt.show() 

# %% id="IkouPeL-1P2k"
