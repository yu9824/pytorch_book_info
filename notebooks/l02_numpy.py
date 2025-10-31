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

# %% [markdown] id="Ap9nWerfEgID"
# ## 講座2 NumPy入門

# %% [markdown] id="COwRNNgpEgIF"
# ### ライブラリインポート

# %% id="Yb2nr4OvEgIG"
# ライブラリのインポート
import numpy as np

# numpyの浮動小数点の表示精度
np.set_printoptions(suppress=True, precision=5)

# %% [markdown] id="a2dIpcJDEgIH"
# ### 定義

# %% [markdown] id="SUcqJQvqEgII"
# #### array関数による定義

# %% id="MdantHK8EgII"
# array関数によるベクトル(1階配列)変数の定義
n1 = np.array([1, 2, 3, 4, 5, 6, 7])

# 結果確認
print(n1)

# 要素数確認
print(n1.shape)

# もう1つの方法
print(len(n1))

# %% id="jVOAlp4qEgIJ"
# array関数による行列(2階配列)変数の定義
n2 = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9],
    [10,11,12]
])

# 結果確認
print(n2)

# 要素数確認
print(n2.shape)

# もう1つの方法
print(len(n2))

# %% [markdown] id="5g065R_PEgIL"
# #### zeros関数、ones関数などの利用

# %% id="L0mDzwgzEgIM"
# zeros関数ですべての要素=0のベクトルを定義
n3 = np.zeros(5)

# 結果確認
print(n3)

# 要素数確認
print(n3.shape)

# %% id="SNslYCarEgIN"
# ones関数ですべての要素=1の行列を定義
n4 = np.ones((2,3))

# 結果確認
print(n4)

# 要素数確認
print(n4.shape)

# %% id="E4K-e3i8EgIN"
# すべての要素が正規分布変数の3階配列
n5 = np.random.randn(2,3,4)

# 結果確認
print(n5)

# 要素数確認
print(n5.shape)

# %% [markdown] id="2nPMkLk4EgIN"
# #### グラフ描画用数値配列の生成

# %% id="znrBwEK3EgIO"
# linspace関数によるグラフ描画用数値配列

# 等間隔に点を取る
# 点の数が第3の引数
# (植木算になる点に注意)
n6 = np.linspace(-1, 1, 11)

# 結果確認
print(n6)

# %% id="w1VY-Hs1EgIO"
# arange関数によるグラフ描画用数値配列

# 等間隔に点を取る
# 間隔値が第3の引数
# (第2引数はmaxでなく「未満」であることに注意)
n7 = np.arange(-1, 1.2, 0.2)

# 結果確認
print(n7)

# %% [markdown] id="SWnpp5hvEgIP"
# ### 操作

# %% [markdown] id="6z1QNbEREgIP"
# #### 特定行・列の抽出

# %% id="b5HuJD8KEgIP"
# 基の変数
print(n2)

# すべての行、0列を抽出
n8 = n2[:,0]
print(n8)

# %% id="9oLq7IFyEgIQ"
# 1行目と3行目を抽出
# True / False の要素の配列で指定する
n2_index = np.array([False, True, False, True])
n9 = n2[n2_index]
print(n9)

# %% [markdown] id="j1nUl3iVEgIQ"
# #### reshape関数

# %% id="7njwM8n7EgIQ"
# 初期変数 1階配列
n10 = np.array(range(24))

# 結果確認
print(n10)

# %% id="9fXWDyy7EgIQ"
# 3x8の2階配列に変形
n11 = n10.reshape(3,8)

# 結果確認
print(n11)

# %% id="-FibSl5vEgIQ"
# -1を指定すると自動計算
n12 = n10.reshape(2, -1, 4)

# 結果確認
print(n12.shape)

# %% id="-XBMkTvcEgIR"
# ベクトルを1行n列の行列に変形

# 基の変数(1階配列)
print(n10.shape)

# 変形
n13 = n10.reshape(1, -1)

# 結果確認
print(n13.shape)

# %% [markdown] id="WFvqwzWTEgIR"
# #### 軸の入れ替え

# %% id="AxyR700LEgIR"
# 転置行列
print(n2)

n14 = n2.T
print(n14)

# %% id="BzGVkgQmEgIR"
# 軸の順番を入れ替える

# 基の変数
print(n12.shape)
print(n12)

# 軸を(1,2,0)の順に入れ替える
n15 = np.transpose(n12, (1, 2, 0))

# 結果確認
print(n15.shape)
print(n15)

# %% [markdown] id="5HAvPD2NEgIR"
# #### 行列の連結

# %% id="pAP98ljYEgIS"
# 操作元の配列

# 2行3列の行列
n16 = np.array(range(1,7)).reshape(2,3)
n17 = np.array(range(7,13)).reshape(2,3)

# 3要素のベクトル
n18 = np.array(range(14,17))

# 2要素のベクトル
n19 = np.array(range(17,19))

print(n16)
print(n17)
print(n18)
print(n19)

# %% id="oqFwewSEEgIS"
# 行列同士の縦連結
n20 = np.vstack([n16, n17])
print(n20)

# 行列とベクトル間の縦連結
n21 = np.vstack([n16, n18])
print(n21)

# %% id="LhU2DzWWEgIS"
# 行列同士の横連結
n22 = np.hstack([n16, n17])
print(n22)

# 行列とベクトル間の横連結
# ベクトルのshopeを(N, 1)形式にする
n23 = n19.reshape(-1, 1)
n24 = np.hstack([n16, n23])
print(n24)

# %% [markdown] id="eTxFh2vPEgIS"
# ### 演算

# %% [markdown] id="pt9dfXlzEgIS"
# #### NumPy変数間の演算

# %% id="2S7zS5YJEgIS"
# 演算元変数
print(n16)
print(n17)

# 行列間の演算
n25 = n16 + n17

# 結果の確認
print(n25)

# %% [markdown] id="Bf0mUbMaEgIT"
# #### ブロードキャスト機能

# %% id="nHBLMIykEgIT"
# ブロードキャスト機能

# 演算元変数
print(n1)

# すべての要素から同じ値を引く
n22 = n1 - 4

# 結果確認
print(n22)

# %% [markdown] id="e7vqyZiNEgIT"
# #### ユニバーサル関数

# %% id="2s4e4assEgIT"
# xの配列の準備
x = np.linspace(0, 2*np.pi, 25)
print(x)

# y=sin(x)の計算
y = np.sin(x)
print(y)

# %% [markdown] id="M8iH6R6AEgIT"
# #### 集約関数

# %% id="4WWIJVodEgIT"
# 集約関数

print(f'元の変数: {n1}')

# 和の計算 sum関数
n23 = np.sum(n1)
print(f'和: {n23}')

# 平均の計算 mean関数
n24 = np.mean(n1)
print(f'平均: {n24}')

# 最大の計算 max関数
n25 = np.max(n1)
print(f'最大値: {n25}')

# 最小の計算 min関数
# こういう書き方もできる
n26 = n1.min()
print(f'最小値: {n26}')

# %% [markdown] id="Lxc0MxN2EgIT"
# ### 応用例

# %% [markdown] id="oWac4vXAEgIT"
# #### 2つの変数の一致数から精度計算

# %% id="YAdcaFTuEgIU"
# 二つのNumPy配列の準備
# 正解値
yt = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
# 予測値
yp = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1])

# 内容表示
print(yt)
print(yp)

# %% id="YPZpBgd5EgIU"
# 配列の各要素を同時に比較する
matched = (yt == yp)
print(matched)

# %% id="eo1d5pyTEgIU"
# 更にこの結果にsum関数をかける
#.対象変数がブーリアン型の場合 True→1 False→0に変換される
# 正解数のカウント方法
correct = matched.sum()

# 全体数はlen(n21)で計算可能
total = len(n21)

# 精度の計算
accuracy = correct/total

print(f'正解数:{correct} 全体数:{total} 精度:{accuracy:.3f}')

# %% [markdown] id="oOcK6XiPEgIU"
# #### ベクトルの数値を一次関数で変換して[0, 1]の範囲に収まるようにする

# %% id="G-fodn3XEgIU"
# すべての値を[0, 1]の範囲に入るように変換する

# 元変数
print(n1)

# 最大値と最小値を集約関数で取得
n1_max = n1.max()
n1_min = n1.min()
print(n1_max, n1_min)

# %% id="8v1-Y2huEgIU"
# 変換(ブロードキャスト機能の利用)
n27 = (n1 - n1_min) / (n1_max - n1_min)
print(n27)

# %% [markdown] id="BKhWnu_eEgIU"
# #### ある条件を満たす行の抽出

# %% id="ecZRkOlMEgIU"
# 「n2の0列が偶数」を判断

# 元変数
print(n2)

n28 = n2[:,0] % 2 == 0

# 結果確認
print(n28)

# %% id="FOPZEmFxEgIU"
# n28がTrueの行のみを抽出

n29 = n2[n28]

# 結果確認
print(n29)

# %% id="JNnJ8SjoEgIV"
