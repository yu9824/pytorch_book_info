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

# %% [markdown] id="y-enheccNr_a"
# # 2章 PyTorch入門

# %% id="vViOXOAVNr_d"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1

# %% id="jfMls7nCNr_e"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %% id="pU48DdrNNr_e"
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# %% [markdown] id="9rmD_Xk2Nr_f"
# ## 2.2 テンソル

# %% [markdown] id="IB73VStvNr_f"
# ### ライブラリインポート

# %% id="VO_-PyPRNr_f"
# PyTorchライブラリ

import torch

# %% [markdown] id="vc5dvrZgNr_f"
# ### いろいろな階数のTensorを作る

# %% id="4I1RggAmNr_g"
# 0階テンソル (スカラー)
r0 = torch.tensor(1.0).float()

# typeを調べる
print(type(r0))

# dtypeを調べる
print(r0.dtype)

# %% id="-VgfrubLNr_g"
# shapeを調べる
print(r0.shape)

# データを調べる
print(r0.data)

# %% id="X0D_1LNKNr_g"
# 1階テンソル (ベクトル)

# 1階のNumPy変数作成
r1_np = np.array([1, 2, 3, 4, 5])
print(r1_np.shape)

# NumPyからテンソルに変換
r1 = torch.tensor(r1_np).float()

# dtypeを調べる
print(r1.dtype)

# shapeを調べる
print(r1.shape)

# データを調べる
print(r1.data)

# %% id="VamMmZJYNr_h"
# 2階テンソル (行列)

# 2階のNmPy変数作成
r2_np = np.array([[1, 5, 6], [4, 3, 2]])
print(r2_np.shape)

# NumPyからテンソルに変換
r2 = torch.tensor(r2_np).float()

# shapeを調べる
print(r2.shape)

# データを調べる
print(r2.data)

# %% id="2X7xX5adNr_h"
# ３階テンソル

# 乱数seedの初期化
torch.manual_seed(123)

# shape=[3,2,2]の正規分布変数テンソルを作る
r3 = torch.randn((3, 2, 2))

# shapeを調べる
print(r3.shape)

# データを調べる
print(r3.data)

# %% id="bUB69bIeNr_h"
# 4階テンソル

# shape=[2,3,2,2]の要素がすべて1のテンソルを作る
r4 = torch.ones((2, 3, 2, 2))

# shapeを調べる
print(r4.shape)

# データを調べる
print(r4.data)

# %% [markdown] id="f9-sP0S1Nr_i"
# ### 整数型テンソルを作る

# %% id="R-0ZFcwrNr_i"
r5 = r1.long()

# dtype　を確認
print(r5.dtype)

# 値を確認
print(r5)

# %% [markdown] id="xBG7lLjANr_i"
# ### view関数

# %% id="LPPH3l5sNr_i"
# 2階化
# 要素数に-1を指定すると、この数を自動調整する

r6 = r3.view(3, -1)

# shape確認
print(r6.shape)

# 値確認
print(r6.data)

# %% id="BXeVtrWVNr_i"
# 1階化
# 要素数に-1を指定すると、この数を自動調整する

r7 = r3.view(-1)

# shape確認
print(r7.shape)

# 値確認
print(r7.data)

# %% [markdown] id="Xv5pRJK1Nr_j"
# ### それ以外の属性

# %% id="5ZYg-dcENr_j"
# requires_grad属性
print('requires_grad: ', r1.requires_grad)

# device属性
print('device: ', r1.device)

# %% [markdown] id="0g-6LRsnNr_j"
# ### item関数

# %% id="Tu3wXoDFNr_j"
# スカラーテンソル(0階テンソル)に対してはitem関数で値を取り出せる

item = r0.item()

print(type(item))
print(item)

# %% id="zwi8v5C0Nr_j"
# 0階以外のテンソルにitem関数は無効

print(r1.item())

# %% id="dvE5Cy3iNr_k"
# 要素数が1つだけの1階テンソルはOK
# (2階以上でも同様)
t1 = torch.ones(1)

# shape確認
print(t1.shape)

# item関数呼び出し
print(t1.item())

# %% [markdown] id="JnrnIuI-Nr_k"
# ### max関数

# %% id="bwyLGh_SNr_k"
# 元テンソルr2の確認
print(r2)

# max関数を引数なしで呼び出すと、全体の最大値が取得できる
print(r2.max())

# %% id="kLkNmIS9Nr_k"
# torch.max関数
# 2つめの引数はどの軸で集約するかを意味する
print(torch.max(r2, 1))

# %% id="lwiblU1vNr_l"
# 何番目の要素が最大値をとるかは、indicesを調べればいい
# 以下の計算は、多値分類で予測ラベルを求めるときによく利用されるパターン
print(torch.max(r2, 1)[1])

# %% [markdown] id="W-CE04b7Nr_l"
# ### NumPy変数への変換

# %% id="WPimA-EMNr_l"
# NumPy化
r2_np = r2.data.numpy()

# type 確認
print(type(r2_np))

# 値確認
print(r2_np)

# %% [markdown] id="85on1hQ4Nr_l"
# ## 2.4 ２次関数の勾配計算

# %% [markdown] id="YiUz1baKNr_l"
#  ### データ準備

# %% id="2K1rvdhiNr_l"
# xをnumpy配列で定義
x_np = np.arange(-2, 2.1, 0.25)

# xの値表示
print(x_np)

# %% id="H_YBx-t9Nr_m"
# (1) 勾配計算用変数の定義
x = torch.tensor(x_np, requires_grad=True, 
    dtype=torch.float32)

# 結果確認
print(x)

# %% [markdown] id="oos9Q69GNr_m"
# ### ２次関数の計算

# %% id="ky1X5hPHNr_m"
# 2次関数の計算
# 裏で計算グラフが自動生成される

y = 2 * x**2 + 2

# %% [markdown] id="iiL8-98INr_m"
# $ y = 2x^2 + 2$ を意味する

# %% id="m0Pwm9jUNr_m"
# yの計算結果確認

print(y)

# %% id="-ONyC7tSNr_m"
# グラフ描画

plt.plot(x.data, y.data)
plt.show()

# %% id="L9S2STdKNr_n"
# 勾配計算のため、sum 関数で 1階テンソルの関数値をスカラー化する
# (sum 関数を各要素で偏微分した結果は1なので、元の関数の微分結果を取得可能 ) 
# ( 詳細はサポートサイトの解説を参照のこと )

z = y.sum()

# %% id="HcLI2RtkNr_n"
# (3) 計算グラフの可視化

# 必要ライブラリのインポート
from torchviz import make_dot

# 可視化関数の呼び出し
g= make_dot(z, params={'x': x})
display(g)

# %% id="Cugbjt5WNr_n"
# (4) 勾配計算

z.backward()

# %% id="Po4-vTPPNr_n"
# (5) 勾配値の取得

print(x.grad)

# %% id="ZaZ5KFSINr_o"
# 元の関数と勾配のグラフ化

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% [markdown] id="m5VKJUI0Nr_o"
# 元の関数が2次関数なので、勾配計算の結果が直線になるのは、妥当な結果

# %% [markdown] id="nrBbdRIvNr_o"
# ここでもう一度勾配計算をしてみる。

# %% id="dOHxzxi0Nr_o"
# 勾配の初期化せずに２度目の勾配計算

y = 2 * x**2 + 2
z = y.sum()
z.backward()

# xの勾配確認
print(x.grad)

# %% [markdown] id="MFsyjUPXNr_o"
# 勾配値は、勾配計算の結果がどんどん加算されてしまう。そのため新しい値を計算したい場合、勾配値のリセットが必要。

# %% id="FURKt3aENr_p"
# (6) 勾配の初期化は関数 zero_()を使う

x.grad.zero_()
print(x.grad)

# %% [markdown] id="F-TnTbjtNr_p"
# ## 2.5 シグモイド関数の勾配計算

# %% [markdown] id="rCQYN3jGNr_p"
# シグモイド関数は数式で表すと次の形になるが今回はPyTorchで提供されている関数を利用する  
# $ y = \dfrac{1}{1 + \exp{(-x)}} $

# %% id="sl8mxsr7Nr_p"
# シグモイド関数の定義
sigmoid = torch.nn.Sigmoid()

# %% id="tT7wMVDuNr_p"
# (2) yの値の計算

y = sigmoid(x)

# %% id="ssP2qjhGNr_p"
# グラフ描画

plt.plot(x.data, y.data)
plt.show()

# %% id="XMXYmt8WNr_q"
# 勾配計算のためには、最終値はスカラーの必要があるため、ダミーでsum関数をかける

z = y.sum()

# %% id="w3H-6V13Nr_q"
# (3) 計算グラフの可視化

g = make_dot(z, params={'x': x})
display(g)

# %% id="zNpm70XeNr_q"
# (4) 勾配計算
z.backward()

# (5) 勾配値の確認
print(x.grad)

# %% id="cyt_foUvNr_q"
# 元の関数と勾配のグラフ化

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% [markdown] id="0kLX9BV9Nr_q"
# シグモイド関数の勾配は、$y(1-y)$になる。  
# 2次関数なので、$y=\dfrac{1}{2}$の時(x=0の時)最大値$\dfrac{1}{4}$を取る。  
# 上のグラフは、この計算結果と一致している。  

# %% id="d6lbqTXUNr_q"
# (6) 勾配の初期化は関数 zero_()を使う

x.grad.zero_()
print(x.grad)


# %% [markdown] id="7tGBthh4Nr_r"
# ### (参考)シグモイド関数を独自に実装した場合

# %% id="Uofxv02YNr_r"
# シグモイド関数の定義

def sigmoid(x):
    return(1/(1 + torch.exp(-x)))


# %% id="1A2ozRy4Nr_r"
# (2) yの値の計算

y = sigmoid(x)

# %% id="vaCaLvR5Nr_r"
# グラフ描画

plt.plot(x.data, y.data)
plt.xlabel('x')
plt.ylabel('y')
plt.title('シグモイド関数のグラフ')
plt.show()

# %% id="8HAx2pNbNr_r"
# 勾配計算のためには、最終値はスカラーの必要があるため、ダミーでsum関数をかける

z = y.sum()

# %% id="m6FR7tA1Nr_r"
# (3) 計算グラフの可視化

params = {'x': x}
g = make_dot(z, params=params)
display(g)

# %% id="9nHpZIC3Nr_s"
# (4) 勾配計算
z.backward()

# (5) 勾配値の確認
print(x.grad)

# %% id="ip_6cFLmNr_s"
# 元の関数と勾配のグラフ化

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% id="t-hoxTV7Nr_s"
