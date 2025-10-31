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
# # 「Pytorch&深層学習プログラミング」２章補足

# %% id="vViOXOAVNr_d" colab={"base_uri": "https://localhost:8080/"} outputId="674c1d46-1e77-4931-cb2f-80a77d4ab1c6"
# 必要ライブラリの導入

# !pip install torchviz | tail -n 1

# %% id="jfMls7nCNr_e"
# 必要ライブラリのインポート

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchviz import make_dot

# %% id="pU48DdrNNr_e"
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# %% id="eQCofFwkn6w0"
# xをnumpy配列で定義
x_np = np.arange(-2, 2.1, 0.25)

# %% [markdown] id="85on1hQ4Nr_l"
# ## sum関数を使った場合

# %% colab={"base_uri": "https://localhost:8080/", "height": 923} id="BfJJtKthn2jx" outputId="7c91954b-2fb6-44a9-b205-864b6719f753"
# 勾配計算用変数の定義
x = torch.tensor(x_np, requires_grad=True, 
    dtype=torch.float32)

# 2次関数の計算
# 裏で計算グラフが自動生成される
y = 2 * x**2 + 2

# 勾配計算のためには、最終値はスカラーの必要があるためsum関数をかける
z = y.sum()

# 可視化関数の呼び出し
g= make_dot(z, params={'x': x})
display(g)

# 勾配計算
z.backward()

# 勾配値の取得
print('勾配値', x.grad)

# 元の関数と勾配のグラフ化
plt.figure(figsize=(6,6))
plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% [markdown] id="2hJbMDXqocZa"
# ## mean関数を使った場合

# %% colab={"base_uri": "https://localhost:8080/", "height": 940} id="8qbzB3iiS2sf" outputId="21b20bac-f33d-47d5-b7fc-edb053ccd694"
# 勾配計算用変数の定義
x = torch.tensor(x_np, requires_grad=True, 
    dtype=torch.float32)

# 2次関数の計算
# 裏で計算グラフが自動生成される
y = 2 * x**2 + 2

# 勾配計算のためには、最終値はスカラーの必要があるため、関数をかける
z = y.mean()

# 可視化関数の呼び出し
g= make_dot(z, params={'x': x})
display(g)

# 勾配計算
z.backward()

# 勾配値の取得
print('勾配値', x.grad)

# 元の関数と勾配のグラフ化
plt.figure(figsize=(6,6))
plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% [markdown] id="IKKj80s5onJa"
# ## max関数を使った場合

# %% id="s6pN1ALYS_pH" outputId="647edc29-e3c7-4dd7-89ea-90c0cfa2fff1" colab={"base_uri": "https://localhost:8080/", "height": 923}
# 勾配計算用変数の定義
x = torch.tensor(x_np, requires_grad=True, 
    dtype=torch.float32)

# 2次関数の計算
# 裏で計算グラフが自動生成される
y = 2 * x**2 + 2

# 勾配計算のためには、最終値はスカラーの必要があるため、関数をかける
z = y.max()


# 可視化関数の呼び出し
g= make_dot(z, params={'x': x})
display(g)

# 勾配計算
z.backward()

# 勾配値の取得
print('勾配値', x.grad)

# 元の関数と勾配のグラフ化
plt.figure(figsize=(6,6))
plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% [markdown] id="k3OZnVbqqN59"
# ## sum関数を使わず、ループを回して各要素を加算した場合

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6zZ-YLDfqMut" outputId="3718e074-8c44-4131-caca-81d4afd99594"
# 勾配計算用変数の定義
x = torch.tensor(x_np, requires_grad=True, 
    dtype=torch.float32)

# 2次関数の計算
# 裏で計算グラフが自動生成される
y = 2 * x**2 + 2

# 勾配計算のためには、最終値はスカラーの必要があるためsum関数をかける
z = torch.tensor(0.0)
for y1 in y:
    z += y1

# 可視化関数の呼び出し
g= make_dot(z, params={'x': x})
display(g)

# 勾配計算
z.backward()

# 勾配値の取得
print('勾配値', x.grad)

# 元の関数と勾配のグラフ化
plt.figure(figsize=(6,6))
plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# %% id="t-hoxTV7Nr_s"
