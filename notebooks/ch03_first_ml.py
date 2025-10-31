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

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/yu9824/pytorch_book_info/blob/main/notebooks/ch03_first_ml.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="oG_Ih-LJWVQe"
# # 3章　初めての機械学習

# %% id="AbiVE7SfWVQh" colab={"base_uri": "https://localhost:8080/"} outputId="83124e8d-e585-47db-a07e-2f0a8412553d"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1

# %% id="qdBSAV7jWVQi"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %% id="qY_7aVQMWVQi"
# PyTorch関連ライブラリ
import torch
from torchviz import make_dot

# %% id="2zhf9g-hWVQi"
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの浮動小数点の表示精度
np.set_printoptions(suppress=True, precision=4)

# %% id="gA7kd5H9kPxI"
# warning表示off
import warnings
warnings.simplefilter('ignore')


# %% [markdown] id="iGBUENgrWVQj"
# ## 3.4 勾配降下法の実装

# %% id="wPlWOAXPWVQj"
def L(u, v):
    return 3 * u**2 + 3 * v**2 - u*v + 7*u - 7*v + 10
def Lu(u, v):
    return 6* u - v + 7
def Lv(u, v):
    return 6* v - u - 7

u = np.linspace(-5, 5, 501)
v = np.linspace(-5, 5, 501)
U, V = np.meshgrid(u, v)
Z = L(U, V)

# %% id="gfvzBvD9WVQj"
# 勾配降下法のシミュレーション
W = np.array([4.0, 4.0])
W1 = [W[0]]
W2 = [W[1]]
N = 21
alpha = 0.05
for i in range(N):
    W = W - alpha *np.array([Lu(W[0], W[1]), Lv(W[0], W[1])])
    W1.append(W[0])
    W2.append(W[1])

# %% id="fnaOfU2mWVQk" colab={"base_uri": "https://localhost:8080/", "height": 661} outputId="9be269c6-c12d-4aad-fa10-2dcc411e3ff6"
n_loop=11

WW1 = np.array(W1[:n_loop])
WW2 = np.array(W2[:n_loop])
ZZ = L(WW1, WW2)
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_zlim(0,250)
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('loss')
ax.view_init(50, 240)
ax.xaxis._axinfo["grid"]['linewidth'] = 2.
ax.yaxis._axinfo["grid"]['linewidth'] = 2.
ax.zaxis._axinfo["grid"]['linewidth'] = 2.
ax.contour3D(U, V, Z, 100, cmap='Blues', alpha=0.7)
ax.plot3D(WW1, WW2, ZZ, 'o-', c='k', alpha=1, markersize=7)
plt.show()

# %% [markdown] id="IagA2uPTWVQk"
# ## 3.5 データ前処理
# 5人の人の身長と体重のデータを使う。  
# 1次関数で身長から体重を予測する場合、最適な直線を求めることが目的。

# %% id="2cnCsxQTWVQk" colab={"base_uri": "https://localhost:8080/"} outputId="e2c00f52-4793-467d-cba2-0d29299dc212"
# サンプルデータの宣言
sampleData1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0,60.1]
])
print(sampleData1)

# %% id="B6ie37XyWVQk"
# 機械学習モデルで扱うため、身長だけを抜き出した変数xと
# 体重だけを抜き出した変数yをセットする

x = sampleData1[:,0]
y = sampleData1[:,1]

# %% id="jTIpXh0AWVQl" colab={"base_uri": "https://localhost:8080/", "height": 580} outputId="a4c53c39-5a86-497a-a2fd-71c510cea475"
# 散布図表示で状況の確認

plt.scatter(x,  y,  c='k',  s=50)
plt.xlabel('$x$: 身長(cm) ')
plt.ylabel('$y$: 体重(kg)')
plt.title('身長と体重の関係')
plt.show()

# %% [markdown] id="_234fjkjWVQl"
# ### 座標系の変換
# 機械学習モデルでは、データは0に近い値を持つことが望ましい。  
# そこで、x, y ともに平均値が0になるように平行移動し、新しい座標系をX, Yとする。

# %% id="lCP56aC2WVQl"
X = x - x.mean()
Y = y - y.mean()

# %% id="gKOh_PZcWVQl" colab={"base_uri": "https://localhost:8080/", "height": 576} outputId="d81943bf-83ec-44be-b058-343f415f0af9"
# 散布図表示で結果の確認

plt.scatter(X,  Y,  c='k',  s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('加工後の身長と体重の関係')
plt.show()

# %% [markdown] id="9z4qsfhFWVQm"
# ## 3.6 予測計算

# %% id="b9D0ASQnWVQm"
# XとYをテンソル変数化する

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

# %% id="g9Fyxbb9WVQm" colab={"base_uri": "https://localhost:8080/"} outputId="14b11a4b-7d75-4d61-ed4e-47d397488e37"
# 結果確認

print(X)
print(Y)

# %% id="OVQNoaiPWVQm"
# 重み変数の定義
# WとBは勾配計算をするので、requires_grad=Trueとする

W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()


# %% id="q4HHhAmEWVQm"
# 予測関数は一次関数

def pred(X):
    return W * X + B


# %% id="3xEp5gkLWVQm"
# 予測値の計算

Yp =  pred(X)

# %% id="bwhCnsfaWVQn" colab={"base_uri": "https://localhost:8080/"} outputId="eb4ca2a4-6a22-44b1-c7e8-10048a187355"
# 結果標示

print(Yp)

# %% id="fZiEnFwGWVQn" colab={"base_uri": "https://localhost:8080/", "height": 395} outputId="2931a0fb-1f7c-411d-ff7d-a0f501b4ff0e"
# 予測値の計算グラフ可視化

params = {'W': W, 'B': B}
g = make_dot(Yp, params=params)
display(g)


# %% [markdown] id="oCQtvOFIWVQn"
# ## 3.7 損失計算

# %% id="cF2BCiT8WVQn"
# 損失関数は誤差二乗平均

def mse(Yp, Y):
    loss = ((Yp - Y) ** 2).mean()
    return loss


# %% id="x3kkCiw3WVQo"
# 損失計算

loss = mse(Yp, Y)

# %% id="1on3xqQOWVQo" colab={"base_uri": "https://localhost:8080/"} outputId="6dcaa82b-712f-4cb1-a0df-9a304c423c71"
# 結果標示

print(loss)

# %% id="zMX3tYAgWVQo" colab={"base_uri": "https://localhost:8080/", "height": 615} outputId="ce2992be-ce11-42d6-ef7b-6efaba20e816"
# 損失の計算グラフ可視化

params = {'W': W, 'B': B}
g = make_dot(loss, params=params)
display(g)

# %% [markdown] id="txlLD4dzWVQo"
# ## 3.8 勾配計算

# %% id="mu7g47KbWVQo"
# 勾配計算

loss.backward()

# %% id="2D-kcOFsWVQo" colab={"base_uri": "https://localhost:8080/"} outputId="5d9ed573-2592-4090-a534-c3dc335c68b9"
# 勾配値確認

print(W.grad)
print(B.grad)

# %% [markdown] id="KZhFz91XWVQp"
# ## 3.9 パラメータ修正

# %% id="x6MvVkl2WVQp"
# 学習率の定義

lr = 0.001

# %% id="0AUYCS2zWVQp" colab={"base_uri": "https://localhost:8080/", "height": 226} outputId="a1388f67-5ff7-4b69-8793-7bdaf84c8912"
#  勾配を元にパラメータ修正

W -= lr * W.grad
B -= lr * B.grad

# %% [markdown] id="jZKWqFp3WVQp"
# WとBは一度計算済みなので、この状態で値の更新ができない  
# 次の書き方にする必要がある

# %% id="mWA2IVKjWVQp"
# 勾配を元にパラメータ修正
# with torch.no_grad() を付ける必要がある

with torch.no_grad():
    W -= lr * W.grad
    B -= lr * B.grad

    # 計算済みの勾配値をリセットする
    W.grad.zero_()
    B.grad.zero_()

# %% id="35hOEh7nWVQp" colab={"base_uri": "https://localhost:8080/"} outputId="ea21d773-126e-4f9f-f3bf-5cc819bdc0f2"
# パラメータと勾配値の確認

print(W)
print(B)
print(W.grad)
print(B.grad)

# %% [markdown] id="Q65yjJyaWVQq"
# 元の値はどちらも1.0だったので、Wは微少量増加、Bは微少量減少したことがわかる。  
# この計算を繰り返すことで、最適なWとBを求めるのが勾配降下法となる。

# %% [markdown] id="gOEcs_OEWVQq"
# ## 3.10 繰り返し計算

# %% id="EGJHgK6QWVQq"
# 初期化

# WとBを変数として扱う
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 繰り返し回数
num_epochs = 500

# 学習率
lr = 0.001

# 記録用配列初期化
history = np.zeros((0, 2))

# %% id="2hKkaSI0WVQq" colab={"base_uri": "https://localhost:8080/"} outputId="36d3859b-036a-4d0f-f14a-b3e28e415f4f"
# ループ処理

for epoch in range(num_epochs):

    # 予測計算
    Yp = pred(X)

    # 損失計算
    loss = mse(Yp, Y)

    # 勾配計算
    loss.backward()

    with torch.no_grad():
        # パラメータ修正
        W -= lr * W.grad
        B -= lr * B.grad

        # 勾配値の初期化
        W.grad.zero_()
        B.grad.zero_()

    # 損失の記録
    if (epoch %10 == 0):
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')


# %% [markdown] id="cPLhlv-1WVQr"
# ## 3.11 結果確認

# %% id="Brcug2NCWVQr" colab={"base_uri": "https://localhost:8080/"} outputId="f09f0b37-436e-4593-e7d0-dd21bca2806c"
# パラメータの最終値
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())

#損失の確認
print(f'初期状態: 損失:{history[0,1]:.4f}')
print(f'最終状態: 損失:{history[-1,1]:.4f}')

# %% id="i3FiFYspWVQr" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="9d2a3f81-f1de-48e0-86ec-c8e72492a2b4"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.show()

# %% [markdown] id="iPiwKDduWVQr"
# ### 散布図に回帰直線を重ね書きする

# %% id="dn2ArxQqWVQr" colab={"base_uri": "https://localhost:8080/"} outputId="3268b33f-8ff0-4813-be0c-1b9bd1f84361"
# xの範囲を求める(Xrange)
X_max = X.max()
X_min = X.min()
X_range = np.array((X_min, X_max))
X_range = torch.from_numpy(X_range).float()
print(X_range)

# 対応するyの予測値を求める
Y_range = pred(X_range)
print(Y_range.data)

# %% id="WvXDYaK5WVQr" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="41292c1a-4d14-4f0e-b0a1-397eb18be820"
# グラフ描画

plt.scatter(X,  Y,  c='k',  s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.plot(X_range.data, Y_range.data, lw=2, c='b')
plt.title('身長と体重の相関直線(加工後)')
plt.show()

# %% [markdown] id="xmt4XrNbWVQs"
# ### 加工前データへの回帰直線描画

# %% id="aFlvzzxaWVQs"
# y座標値とx座標値の計算

x_range = X_range + x.mean()
yp_range = Y_range + y.mean()

# %% id="jjbHJh7cWVQs" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="850bedd4-73a6-4144-e6c4-60810e6c41aa"
# グラフ描画

plt.scatter(x,  y,  c='k',  s=50)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(x_range, yp_range.data, lw=2, c='b')
plt.title('身長と体重の相関直線(加工前)')
plt.show()

# %% [markdown] id="jbY0y976WVQs"
# ## 3.12 最適化関数とstep関数の利用

# %% id="MFBuUEHgWVQs"
# 初期化

# WとBを変数として扱う
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 繰り返し回数
num_epochs = 500

# 学習率
lr = 0.001

# optimizerとしてSGD(確率的勾配降下法)を指定する
import torch.optim as optim
optimizer = optim.SGD([W, B], lr=lr)

# 記録用配列初期化
history = np.zeros((0, 2))

# %% id="DQRx7YQuWVQs" colab={"base_uri": "https://localhost:8080/"} outputId="06b7be7f-9704-47eb-ee08-21a811c68a67"
# ループ処理

for epoch in range(num_epochs):

    # 予測計算
    Yp = pred(X)

    # 損失計算
    loss = mse(Yp, Y)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    #勾配値初期化
    optimizer.zero_grad()

    # 損失値の記録
    if (epoch %10 == 0):
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')

# %% id="IPNHGAdeWVQt" colab={"base_uri": "https://localhost:8080/"} outputId="8e727433-1d3e-4bb1-c2cd-8afe5d9a26e1"
# パラメータの最終値
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())

#損失の確認
print(f'初期状態: 損失:{history[0,1]:.4f}')
print(f'最終状態: 損失:{history[-1,1]:.4f}')

# %% id="Gllq7RCrWVQt" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="395998e2-5577-467b-8e55-d7ebff7df093"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.show()

# %% [markdown] id="8xhELdZxWVQt"
# 3.7の結果と見比べるとまったく同じであることがわかる。  
# つまり、step関数でやっていることは、次のコードと同じ。
#
# ```py3
#
#  with torch.no_grad():
#         # パラメータ修正 (フレームワークを使う場合はstep関数)
#         W -= lr * W.grad
#         B -= lr * B.grad
# ```

# %% [markdown] id="llF8abxMWVQt"
# ### 最適化関数のチューニング

# %% id="PJrzSRW4WVQt"
# 初期化

# WとBを変数として扱う
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 繰り返し回数
num_epochs = 500

# 学習率
lr = 0.001

# optimizerとしてSGD(確率的勾配降下法)を指定する
import torch.optim as optim
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9)

# 記録用配列初期化
history2 = np.zeros((0, 2))

# %% id="zgsU861zWVQt" colab={"base_uri": "https://localhost:8080/"} outputId="af1b48ff-aca7-4e31-8977-fece984bf15f"
# ループ処理

for epoch in range(num_epochs):

    # 予測計算
    Yp = pred(X)

    # 損失計算
    loss = mse(Yp, Y)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    #勾配値初期化
    optimizer.zero_grad()

    # 損失値の記録
    if (epoch %10 == 0):
        item = np.array([epoch, loss.item()])
        history2 = np.vstack((history2, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')


# %% id="e1BTUhNSWVQu" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="25efc97b-4627-40de-8bc6-660bddf3cf64"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b', label='デフォルト設定')
plt.plot(history2[:,0], history2[:,1], 'k', label='momentum=0.9')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.legend()
plt.title('学習曲線(損失)')
plt.show()


# %% [markdown] id="c_Bw4JEnWVQu"
# ## コラム　局所最適解

# %% id="WpLZKllPWVQu"
def f(x):
    return x * (x+1) * (x+2) * (x-2)


# %% id="ieChuekDWVQu" colab={"base_uri": "https://localhost:8080/", "height": 499} outputId="b51cf9ac-15a1-414c-879e-8248dcaa78a6"
x = np.arange(-3, 2.7, 0.05)
y = f(x)

plt.plot(x, y)
plt.axis('off')
plt.show()

# %% id="eDKFJHWpWVQu"
