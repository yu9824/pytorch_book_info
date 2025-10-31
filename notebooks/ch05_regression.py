# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/yu9824/pytorch_book_info/blob/main/notebooks/ch05_regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="FXGXzQQSZhqU"
# # 5章　線形回帰

# %% id="ID01O-IvXpCi" colab={"base_uri": "https://localhost:8080/"} outputId="fc132416-86bf-472e-b50b-2c3cb762ce9f"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1
# !pip install torchinfo | tail -n 1

# %% id="9BuGuNS4XpCk"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %% id="nFqLX4N-XpCk"
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

# %% id="EWZHxbB1XpCl"
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの浮動小数点の表示精度
np.set_printoptions(suppress=True, precision=4)

# %% [markdown] id="aL-55tM8XpCl"
# ## 5.3 線形関数(nn.Linear)

# %% [markdown] id="RyH3OPekXpCl"
# ###  入力:1 出力:1 の線形関数

# %% id="lzycf0HxXpCm" colab={"base_uri": "https://localhost:8080/"} outputId="777bc1a1-d8b5-499b-a662-7d0bab1e938d"
# 乱数の種固定
torch.manual_seed(123)

# 入力:1 出力:1 の線形関数の定義
l1 = nn.Linear(1, 1)

# 線形関数の表示
print(l1)

# %% id="pSvee2CeXpCm" colab={"base_uri": "https://localhost:8080/"} outputId="8ca29058-e137-45d9-d442-ed79fb657762"
# パラメータ名、パラメータ値、shapeの表示

for param in l1.named_parameters():
    print('name: ', param[0])
    print('tensor: ', param[1])
    print('shape: ', param[1].shape)

# %% id="LKYmasnBXpCn" colab={"base_uri": "https://localhost:8080/"} outputId="d092f86f-fb94-46e6-c0ad-fdea1b166e5a"
# 初期値設定
nn.init.constant_(l1.weight, 2.0)
nn.init.constant_(l1.bias, 1.0)

# 結果確認
print(l1.weight)
print(l1.bias)

# %% id="UZNxp3fYXpCn" colab={"base_uri": "https://localhost:8080/"} outputId="28d5feee-f013-42d2-b549-158e6a813c44"
# テスト用データ生成

# x_npをnumpy配列で定義
x_np = np.arange(-2, 2.1, 1)

# Tensor化
x = torch.tensor(x_np).float()

# サイズを(N,1)に変更
# x = x.view(-1,1)
x = x.reshape(-1, 1)

# 結果確認
print(x.shape)
print(x)

# %% id="2De8gINFXpCo" colab={"base_uri": "https://localhost:8080/"} outputId="d8915368-522f-45e1-c8f1-68832c024b12"
# 1次関数のテスト

y = l1(x)

print(y.shape)
print(y.data)

# %% [markdown] id="d0csbLg9XpCo"
# ### 入力:2 出力:1 の線形関数

# %% id="VyebhECaXpCp" colab={"base_uri": "https://localhost:8080/"} outputId="4442ea19-c87d-4ab9-8cc2-a43114e835ec"
# 入力:2 出力:1 の線形関数の定義
l2 = nn.Linear(2, 1)

# 初期値設定
nn.init.constant_(l2.weight, 1.0)
nn.init.constant_(l2.bias, 2.0)

# 結果確認
print(l2.weight)
print(l2.bias)

# %% id="S1yCAOCcXpCp" colab={"base_uri": "https://localhost:8080/"} outputId="78137f39-5213-4c02-856f-4a2e20fb9eed"
# 2次元numpy配列
x2_np = np.array([[0, 0], [0, 1], [1, 0], [1,1]])

# Tensor化
x2 =  torch.tensor(x2_np).float()

# 結果確認
print(x2.shape)
print(x2)

# %% id="bZ3OsIWJXpCp" colab={"base_uri": "https://localhost:8080/"} outputId="1901e234-8eb8-47c8-b132-39914bff3af2"

# 関数値計算
y2 = l2(x2)

# shape確認
print(y2.shape)

# 値確認
print(y2.data)

# %% [markdown] id="53Ae8ANsXpCq"
# ### 入力:2 出力:3 の線形関数

# %% id="o5xGgx2kXpCq" colab={"base_uri": "https://localhost:8080/"} outputId="94de4a1b-3428-431a-b0a7-a87fc46dd4a2"
# 入力:2 出力:3 の線形関数の定義

l3 = nn.Linear(2, 3)

# 初期値設定
nn.init.constant_(l3.weight[0,:], 1.0)
nn.init.constant_(l3.weight[1,:], 2.0)
nn.init.constant_(l3.weight[2,:], 3.0)
nn.init.constant_(l3.bias, 2.0)

# 結果確認
print(l3.weight.shape, x2.shape)    # 内積のはずだが、内積とは形が違うのでちょっとイメージしづらい。（numpy的には(n, m) @ (m, l)）
print(l3.weight)
print(l3.bias)

# %% id="rL6qApqTXpCq" colab={"base_uri": "https://localhost:8080/"} outputId="8e0ceae7-c129-45f9-f440-232cb85d5ea4"
# 関数値計算
y3 = l3(x2)

# shape確認
print(y3.shape)

# 値確認
print(y3.data)


# %% [markdown] id="6Qy6LqOfRT0E"
# ## 5.4 カスタムクラスを利用したモデル定義

# %% id="3bcQHUdyXpCt"
# モデルのクラス定義

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()

        # 出力層の定義
        self.l1 = nn.Linear(n_input, n_output)

    # 予測関数の定義
    def forward(self, x):
        x1 = self.l1(x) # 線形回帰
        return x1


# %% id="5FLHS8bvXpCt"
# ダミー入力
inputs = torch.ones(100,1)

# インスタンスの生成 (１入力1出力の線形モデル)
n_input = 1
n_output = 1
net = Net(n_input, n_output)

# 予測
outputs = net(inputs)

# %% [markdown] id="KFmZjNAlXpCr"
#
# ## 5.6 データ準備
# UCI公開データセットのうち、回帰でよく使われる「ボストン・データセット」を用いる。
#
# https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
#
# オリジナルのデーセットは、17項目の入力値から、不動産価格を予測する目的のものだが、
# 一番単純な「単回帰モデル」(1入力)のモデルを作るため、このうち``RM``の1項目だけを抽出する。
#

# %% id="wva-PxPSJzyg" colab={"base_uri": "https://localhost:8080/"} outputId="0186331a-bf21-4501-c1e8-acca552417de"
# 学習用データ準備

# 「ボストン・データセット」はscikit-learnのライブラリでも取得できるが、
# その場合、将来版で利用できなくなる予定のため、別Webサイトから取得する
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+",
    skiprows=22, header=None)
x_org = np.hstack([raw_df.values[::2, :],
    raw_df.values[1::2, :2]])
yt = raw_df.values[1::2, 2]
feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT'])

# 結果確認
print('元データ', x_org.shape, yt.shape)
print('項目名: ', feature_names)

# %% id="ro0A3LlSXpCs" colab={"base_uri": "https://localhost:8080/"} outputId="2d43637c-aec4-46f5-8a88-dbe5e527be0f"
# データ絞り込み (項目 RMのみ)
x = x_org[:,feature_names == 'RM']
print('絞り込み後', x.shape)
print(x[:5,:])

# 正解データ yの表示
print('正解データ')
print(yt[:5])

# %% id="DvwFf3Taa-1O" colab={"base_uri": "https://localhost:8080/", "height": 576} outputId="739457a3-c0f2-45da-d01d-bc191aac7759"
# 散布図の表示

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('部屋数')
plt.ylabel('価格')
plt.title('部屋数と価格の散布図')
plt.show()

# %% [markdown] id="Pv-YMbfwXpCs"
# ## 5.7 モデル定義

# %% id="UG4aJY0YbHO7" colab={"base_uri": "https://localhost:8080/"} outputId="0efcf577-1b9e-49f1-d0f2-847366f95f4c"
# 変数定義

# 入力次元数
n_input= x.shape[1]

# 出力次元数
n_output = 1

print(f'入力次元数: {n_input}  出力次元数: {n_output}')


# %% id="Qa_cV8urbs9s"
# 機械学習モデル（予測モデル）クラス定義

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()

        # 出力層の定義
        self.l1 = nn.Linear(n_input, n_output)

        # 初期値を全部1にする
        # 「ディープラーニングの数学」と条件を合わせる目的
        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)

    # 予測関数の定義
    def forward(self, x):
        x1 = self.l1(x) # 線形回帰
        return x1


# %% id="wQ4TtPGQkWql"
# インスタンスの生成
# １入力1出力の線形モデル

net = Net(n_input, n_output)

# %% id="bFf1N0HsXpCu" colab={"base_uri": "https://localhost:8080/"} outputId="34c7d8ab-66ff-4989-924b-85e9d071c353"
# モデル内のパラメータの確認
# モデル内の変数取得にはnamed_parameters関数を利用する
# 結果の第1要素が名前、第2要素が値
#
# predict.weightとpredict.biasがあることがわかる
# 初期値はどちらも1.0になっている

for parameter in net.named_parameters():
    print(f'変数名: {parameter[0]}')
    print(f'変数値: {parameter[1].data}')

# %% id="RjHjfAf7XpCu" colab={"base_uri": "https://localhost:8080/"} outputId="2200f901-ea09-4040-fee0-e8dff97d6291"
# パラメータのリスト取得にはparameters関数を利用する

for parameter in net.parameters():
    print(parameter)

# %% [markdown] id="LTqxnIJaXpCv"
# ### モデル確認

# %% id="_zaPgb2td6vV" colab={"base_uri": "https://localhost:8080/"} outputId="8b66bb73-4976-41e7-abb7-4d1ecf8724f4"
# モデルの概要表示

print(net)

# %% id="uWXVbu0leJgB" colab={"base_uri": "https://localhost:8080/"} outputId="ebb9287a-4889-4115-a069-2a00fc178738"
# モデルのサマリー表示

from torchinfo import summary
summary(net, (1,))

# %% [markdown] id="aSx3Ha0cXpCv"
# ### 損失関数と最適化関数

# %% id="_-8Dq5hWfeoB"
# 損失関数： 平均2乗誤差
criterion = nn.MSELoss()

# 学習率
lr = 0.01

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)


# %% [markdown] id="7FXFabMEXpCw"
# ## 5.8 勾配降下法

# %% id="-hiUPe-uXpCw" colab={"base_uri": "https://localhost:8080/"} outputId="e22bb376-2f71-4634-8a5b-bf63ea849208"
# 入力変数x と正解値 ytのテンソル変数化

inputs = torch.tensor(x).float()
labels = torch.tensor(yt).float()

# 次元数確認

print(inputs.shape)
print(labels.shape)

# %% id="0iRTGvM4XpCw" colab={"base_uri": "https://localhost:8080/"} outputId="a69128e3-76f3-406d-a9cb-b6215c26772d"
# 損失値計算用にlabels変数を(N,1)次元の行列に変換する

labels1 = labels.view((-1, 1))

# 次元数確認
print(labels1.shape)

# %% id="foVlKfQ5XpCw"
# 予測計算

outputs = net(inputs)

# %% id="frF6g1MhXpCx" colab={"base_uri": "https://localhost:8080/"} outputId="e191c70d-305c-4568-8b36-8bcc38860894"

#  損失計算
loss = criterion(outputs, labels1)

# 損失値の取得
print(f'{loss.item():.5f}')

# %% id="bHjWn0NfXpCx" colab={"base_uri": "https://localhost:8080/", "height": 469} outputId="14cee67a-031f-4bd7-995e-cec33113719e"

# 損失の計算グラフ可視化

g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# %% id="j6IUY2obXpCx" colab={"base_uri": "https://localhost:8080/"} outputId="351feaf8-412b-4ddf-bafa-ca0ee8403c30"
# 予測計算
outputs = net(inputs)

#  損失計算
loss = criterion(outputs, labels1)

# 勾配計算
loss.backward()

# 勾配の結果が取得可能に
print(net.l1.weight.grad)
print(net.l1.bias.grad)

# %% id="SMwYMMXAXpCy" colab={"base_uri": "https://localhost:8080/"} outputId="d480b084-ea46-4d54-bfae-c78dae10f52d"
# パラメータ修正
optimizer.step()

# パラメータ値が変わる
print(net.l1.weight)
print(net.l1.bias)

# %% id="4SMBbG19XpCy" colab={"base_uri": "https://localhost:8080/"} outputId="b6b76961-8ae7-4dec-8b16-5474b0b6e287"
# 勾配値の初期化
optimizer.zero_grad()

# 勾配値がすべてゼロになっている
print(net.l1.weight.grad)
print(net.l1.bias.grad)

# %% [markdown] id="qGe0W7syXpCy"
# ### 繰り返し計算

# %% id="s6sIrMqiXpCy"
# 学習率
lr = 0.01

# インスタンス生成　(パラメータ値初期化)
net = Net(n_input, n_output)

# 損失関数： 平均2乗誤差
criterion = nn.MSELoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 50000

# 評価結果記録用 (損失関数値のみ記録)
history = np.zeros((0,2))

# %% id="7Db7l67DeT9-" colab={"base_uri": "https://localhost:8080/"} outputId="38bc8f69-470b-4d0d-efdb-36a335b2dc56"
# 繰り返し計算メインループ

for epoch in range(num_epochs):

    # 勾配値初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 損失計算
    # 「ディープラーニングの数学」に合わせて2で割った値を損失とした
    loss = criterion(outputs, labels1) / 2.0

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    # 100回ごとに途中経過を記録する
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

# %% [markdown] id="UNDxax-lXpC2"
# ## 5.9 結果確認

# %% id="kWRH7ExmjB8B" colab={"base_uri": "https://localhost:8080/"} outputId="9ebd4047-0548-4da4-8568-92ef1512cf06"
# 損失初期値と最終値

print(f'損失初期値: {history[0,1]:.5f}')
print(f'損失最終値: {history[-1,1]:.5f}')

# %% id="LUh7GLCSfQNJ" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="30af0602-6876-4934-8328-c4a87f796ee9"

# 学習曲線の表示 (損失)
# 最初の1つを除く

plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.show()

# %% id="SboqNthnmTKf" colab={"base_uri": "https://localhost:8080/"} outputId="90434d77-a484-4220-f967-b0625b7d270f"
# 回帰直線の算出

# xの最小値、最大値
xse = np.array((x.min(), x.max())).reshape(-1,1)
Xse = torch.tensor(xse).float()

with torch.no_grad():
  Yse = net(Xse)

print(Yse.numpy())

# %% id="JNynfK4pngwe" colab={"base_uri": "https://localhost:8080/", "height": 576} outputId="2fdf9fc1-c365-4546-f970-a2cd666ef92e"
# 散布図と回帰直線の描画

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('部屋数')
plt.ylabel('価格')
plt.plot(Xse.data, Yse.data, c='k')
plt.title('散布図と回帰直線')
plt.show()

# %% [markdown] id="qwGTzOJbr8Hs"
# ## 5.10 重回帰モデルへの拡張

# %% id="8gAZgC1GpjEV" colab={"base_uri": "https://localhost:8080/"} outputId="3bde64ad-4c89-4905-fdd7-178950b2aefc"
# 列(LSTAT: 低所得者率)の追加

x_add = x_org[:,feature_names == 'LSTAT']
x2 = np.hstack((x, x_add))

# shapeの表示
print(x2.shape)

# 入力データxの表示
print(x2[:5,:])

# %% id="z99b-YPvsM5_" colab={"base_uri": "https://localhost:8080/"} outputId="d400cd67-2de2-44a4-ce5a-04e476c39c27"
# 今度は入力次元数=2

n_input = x2.shape[1]
print(n_input)

# モデルインスタンスの生成
net = Net(n_input, n_output)

# %% id="tvTwYb-gXpC5" colab={"base_uri": "https://localhost:8080/"} outputId="b4c5d9bf-8de2-4ac1-e4ec-2a48f1573871"
# モデル内のパラメータの確認
# predict.weight が2次元に変わった

for parameter in net.named_parameters():
    print(f'変数名: {parameter[0]}')
    print(f'変数値: {parameter[1].data}')

# %% id="5iBFb2A9s2K1" colab={"base_uri": "https://localhost:8080/"} outputId="bce0962c-26da-4052-9386-2151d33daf50"
# モデルの概要表示

print(net)

# %% id="jwdT12mss60n" colab={"base_uri": "https://localhost:8080/"} outputId="15dc828d-7729-488a-fd0b-b9cc8f93e77e"
# モデルのサマリー表示

from torchinfo import summary
summary(net, (2,))

# %% id="U3xw_jxTXpC5" colab={"base_uri": "https://localhost:8080/"} outputId="93664a5b-770e-45ac-8727-2b26ccaf8a3e"
# 入力変数x2 のテンソル変数化
# labels, labels1は前のものをそのまま利用

inputs = torch.tensor(x2).float()
inputs

# %% [markdown] id="dGB9wkr-XpC6"
# ### くり返し計算

# %% id="8cyjWATHXpC6"
# 初期化処理

# 学習率
lr = 0.01

# インスタンス生成　(パラメータ値初期化)
net = Net(n_input, n_output)

# 損失関数： 平均2乗誤差
criterion = nn.MSELoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 50000

# 評価結果記録用 (損失関数値のみ記録)
history = np.zeros((0,2))

# %% id="7rEh-tPpXpC6" colab={"base_uri": "https://localhost:8080/"} outputId="8d3786a4-3214-49f1-82f5-4298b9bf4a86"
# 繰り返し計算メインループ

for epoch in range(num_epochs):

    # 勾配値初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 誤差計算
    # 「ディープラーニングの数学」に合わせて2で割った値を損失とした
    loss = criterion(outputs, labels1) / 2.0

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    # 100回ごとに途中経過を記録する
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

# %% [markdown] id="_SvyW5S2XpC6"
# ## 5.11 学習率の変更

# %% id="wt92VISFtaLj"
# 繰り返し回数
#num_epochs = 50000
num_epochs = 2000

# 学習率
#l r = 0.01
lr = 0.001

# モデルインスタンスの生成
net = Net(n_input, n_output)

# 損失関数： 平均2乗誤差
criterion = nn.MSELoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# %% id="tlAkM8K5t4XV" colab={"base_uri": "https://localhost:8080/"} outputId="54007c27-a692-48b6-c203-3a3467c7cb1e"
# 繰り返し計算メインループ

# 評価結果記録用 (損失関数値のみ記録)
history = np.zeros((0,2))

for epoch in range(num_epochs):

    # 勾配値初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 誤差計算
    loss = criterion(outputs, labels1) / 2.0

    #勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    # 100回ごとに途中経過を記録する
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

# %% id="Y4UHX9Ast8Je" colab={"base_uri": "https://localhost:8080/"} outputId="ee1891b7-294d-4f3e-d03b-9bdc8df0a1c1"
# 損失初期値、最終値

print(f'損失初期値: {history[0,1]:.5f}')
print(f'損失最終値: {history[-1,1]:.5f}')

# %% id="hYPtlO6wuCoy" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="ea5fe208-8a59-4f66-a73e-aa431bb06f0b"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.show()

# %% id="j-CVGgMlXpC8"
S
