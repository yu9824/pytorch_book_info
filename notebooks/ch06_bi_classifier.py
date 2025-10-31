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
# <a href="https://colab.research.google.com/github/yu9824/pytorch_book_info/blob/main/notebooks/ch06_bi_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="RMJ-p2-QPsqR"
# # 6章　2値分類

# %% id="o8f1MNUoPsqa" colab={"base_uri": "https://localhost:8080/"} outputId="5cd748e7-6c46-4534-fa6b-3c10c9cef677"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1
# !pip install torchinfo | tail -n 1

# %% id="hkEhF-rHPsqc"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %% id="fa0xuJKVPsqd"
# torch関連ライブラリのインポート

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot

# %% id="OLGjeQzHPsqd"
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# %% [markdown] id="PqWB_9tDPsqe"
# ## 6.3 シグモイド関数

# %% id="DcYYjzNJPsqe" colab={"base_uri": "https://localhost:8080/", "height": 553} outputId="06b208c3-6deb-4384-8d71-971e37a5c758"
# NumPy配列でxデータを定義
x_np = np.arange(-4, 4.1, 0.25)

# データをTensor形式に変換
x = torch.tensor(x_np).float()

# yの値を計算
y = torch.sigmoid(x)

# グラフ描画
plt.title('シグモイド関数のグラフ')
plt.plot(x.data, y.data)
plt.show()

# %% [markdown] id="nDG_cwuNPsqf"
# ## 6.7 データ準備

# %% id="vhcBelEXPsqf" colab={"base_uri": "https://localhost:8080/"} outputId="cfdfe914-1d04-4edd-d6dc-400437cc4585"
# 学習用データ準備

# ライブラリのインポート
from sklearn.datasets import load_iris

# データ読み込み
iris = load_iris()

# 入力データと正解データ取得
x_org, y_org = iris.data, iris.target

# 結果確認
print('元データ', x_org.shape, y_org.shape)

# %% id="Uo1YPkPEPsqf" colab={"base_uri": "https://localhost:8080/"} outputId="90e21ffd-5e1f-403e-8a8f-032c9daba319"
# データ絞り込み
#   クラス0, 1のみ
#   項目sepal_lengthとsepal_widthのみ

x_data = iris.data[:100,:2]
y_data = iris.target[:100]

# 結果確認
print('対象データ', x_data.shape, y_data.shape)

# %% [markdown] id="_uRG9oXGPsqg"
# ### 訓練データ・検証データの分割

# %% id="YUS4x_tmPsqg" colab={"base_uri": "https://localhost:8080/"} outputId="fbabfee2-78fe-4b47-a8ad-711ecac5c699"
# 　元データのサイズ
print(x_data.shape, y_data.shape)

# 訓練データ、検証データに分割 (シャフルも同時に実施)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30,
    random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %% [markdown] id="sRiAjjbePsqh"
# ### 訓練データの散布図表示

# %% id="zu-SEFS2Psqh" colab={"base_uri": "https://localhost:8080/", "height": 555} outputId="84e78a3a-ff1c-4755-cdad-50a0aefbc0d9"
# 散布図の表示

x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='b', label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='k', label='1 (versicolor)')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()

# %% [markdown] id="CZ3NxWrlPsqh"
# ## 6.8 モデル定義

# %% id="8Je8itWpPsqh" colab={"base_uri": "https://localhost:8080/"} outputId="cdb5964e-67a9-43dd-e87d-07f4937b651f"
# 入力次元数　(今の場合2)
n_input= x_train.shape[1]

# 出力次元数
n_output = 1

# 結果確認
print(f'n_input: {n_input}  n_output:{n_output}')


# %% id="DdDi6zsnPsqi"
# モデルの定義
# 2入力1出力のロジスティック回帰モデル

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()

        # 初期値を全部1にする
        # 「ディープラーニングの数学」と条件を合わせる目的
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    # 予測関数の定義
    def forward(self, x):
        # 最初に入力値を線形関数にかけたを計算する
        x1 = self.l1(x)
        # 計算結果にシグモイド関数をかける
        x2 = self.sigmoid(x1)
        return x2


# %% id="SgP8JkLBPsqi"
# インスタンスの生成

net = Net(n_input, n_output)

# %% [markdown] id="QcmbWo3xPsqj"
# ### モデル確認

# %% id="aWzmDIs_Psqj" colab={"base_uri": "https://localhost:8080/"} outputId="7cb25570-fb4d-496f-96bf-3a927ddc0e14"
# モデル内のパラメータの確認
# l1.weightとl1.biasがあることがわかる

for parameter in net.named_parameters():
    print(parameter)

# %% id="99WSyEiQPsqj" colab={"base_uri": "https://localhost:8080/"} outputId="ebef6a8f-8fa8-414d-bf42-86a9489b0167"
# モデルの概要表示

print(net)

# %% id="SJ34VvxLPsqk" colab={"base_uri": "https://localhost:8080/"} outputId="47653332-a9ab-44f0-c345-aea17b6971fe"
# モデルのサマリー表示

summary(net, (2,))

# %% [markdown] id="2KHOYmE6Psqk"
# ### 最適化アルゴリズムと損失関数

# %% id="Zfv1SnjRPsqk"
# 損失関数： 交差エントロピー関数
criterion = nn.BCELoss()

# 学習率
lr = 0.01

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# %% colab={"base_uri": "https://localhost:8080/"} id="NAAoswhy0ikf" outputId="f7bcbd0a-169e-4354-e686-5f71a9c6ba09"
list(net.parameters())

# %% colab={"base_uri": "https://localhost:8080/"} id="G7hoJxTl0tHW" outputId="150b3523-0634-4d4e-defc-96684f360d88"
dict(net.named_parameters())


# %% [markdown] id="OIEA8RBFPsqk"
# ## 6.9 勾配降下法

# %% id="bPBqrftvPsql"
# 入力データ x_train と正解データ y_train のテンソル化

inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()

# 正解データはN行1列の行列に変換する
labels1 = labels.view((-1,1))

# 検証データのテンソル化
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

# 検証用の正解データもN行1列の行列に変換する
labels1_test = labels_test.view((-1,1))

# %% id="e3TvdQuOPsql" colab={"base_uri": "https://localhost:8080/", "height": 543} outputId="09c6da28-0f7a-45d6-ad45-b101139bd914"
# 予測計算
outputs = net(inputs)

# 損失計算
loss = criterion(outputs, labels1)

# 損失の計算グラフ可視化
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# %% [markdown] id="_ciIvpF8Psql"
# ### 繰り返し計算

# %% id="syR4VfrZPsql"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： 交差エントロピー関数
criterion = nn.BCELoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 10000

# 記録用リストの初期化
history = np.zeros((0,5))

# %% id="50gln2ooPsqm" colab={"base_uri": "https://localhost:8080/"} outputId="5b8581c8-d44a-4cee-d20d-93ecc91ebfac"
# 繰り返し計算メインループ

for epoch in range(num_epochs):
    # 訓練フェーズ

    #勾配値初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 損失計算
    loss = criterion(outputs, labels1)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    # 損失の保存(スカラー値の取得)
    train_loss = loss.item()

    # 予測ラベル(1 or 0)計算
    predicted = torch.where(outputs < 0.5, 0, 1)

    # 精度計算
    train_acc = (predicted == labels1).sum() / len(y_train)

    # 予測フェーズ

    # 予測計算
    outputs_test = net(inputs_test)

    # 損失計算
    loss_test = criterion(outputs_test, labels1_test)

    # 損失の保存（スカラー値の取得）
    val_loss =  loss_test.item()

    # 予測ラベル(1 or 0)計算
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)

    # 精度計算
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# %% [markdown] id="BTk6dHwfPsqm"
# ## 6.10 結果確認

# %% id="6YBWkJXvPsqm" colab={"base_uri": "https://localhost:8080/"} outputId="091bb328-17b4-4c4d-b9aa-6173d7e5f4e7"
#損失と精度の確認

print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# %% id="E_FBeQjBPsqn" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="6efcb0ff-8107-4d8c-96df-9e961f19952b"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b', label='訓練')
plt.plot(history[:,0], history[:,3], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.legend()
plt.show()

# %% id="PsEjWTI2Psqn" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="7da146d7-54d8-4782-9a4f-0a0ece45e10c"
# 学習曲線の表示 (精度)

plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線(精度)')
plt.legend()
plt.show()

# %% [markdown] id="QtiZzCnKPsqn"
# ### 決定境界のグラフ表示

# %% id="ZEV3sKftPsqo"
# 検証データを散布図用に準備

x_t0 = x_test[y_test==0]
x_t1 = x_test[y_test==1]

# %% id="3knK_C7kPsqo" colab={"base_uri": "https://localhost:8080/"} outputId="232b98d0-927f-4709-d062-063435618bd1"
# パラメータの取得

bias = net.l1.bias.data.numpy()
weight = net.l1.weight.data.numpy()
print(f'BIAS = {bias}, WEIGHT = {weight}')

# 決定境界描画用 x1の値から x2の値を計算する
def decision(x):
    return(-(bias + weight[0,0] * x)/ weight[0,1])

# 散布図のx1の最小値と最大値
xl = np.array([x_test[:,0].min(), x_test[:,0].max()])
yl = decision(xl)

# 結果確認
print(f'xl = {xl}  yl = {yl}')

# %% [markdown] id="ekZ4mnj34UtO"
# なぜ上記で決定境界が求められるのかここには書いてなかったので補足。
#
# 決定境界はちょうどそれぞれのクラスになる確率が0.5同士の点の集合なので、
#
# $$
# \frac{1}{2} = \frac{1}{1+\mathrm{exp} \left\{ - \left( w_0 + w_1 x_1 + w_2 x_2 \right) \right\} } \\
# \mathrm{i.e.} \ w_0 + w_1 x_1 + w_2 x_2 = 0 \\
# \therefore x_2 = -\frac{w_0 + w_1 x_1}{w_2}
# $$

# %% id="wV_ncIo0Psqo" colab={"base_uri": "https://localhost:8080/", "height": 550} outputId="f4085825-32ef-4925-a3dd-a03edb203a1f"
# 散布図表示
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x',
        c='b', s=50, label='class 0')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o',
        c='k', s=50, label='class 1')

# 決定境界直線
plt.plot(xl, yl, c='b')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()


# %% [markdown] id="h0jF0iMzPsqo"
# ## コラム　BCELoss関数とBCEWithLogitsLoss関数の違い

# %% id="0yNjht06Psqp"
# モデルの定義
# 2入力1出力のロジスティック回帰モデル

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # 初期値を全部1にする
        # 「ディープラーニングの数学」と条件を合わせる目的
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    # 予測関数の定義
    def forward(self, x):
        # 入力値と行列の積を計算する
        x1 = self.l1(x)
        return x1


# %% id="A3dil0VDPsqp"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： logits付き交差エントロピー関数
criterion = nn.BCEWithLogitsLoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 10000

# 記録用リストの初期化
history = np.zeros((0,5))

# %% id="abFfu6brPsqp"
# 繰り返し計算メインループ

for epoch in range(num_epochs):
    # 訓練フェーズ

    #勾配値初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 損失計算
    loss = criterion(outputs, labels1)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    # 損失のスカラー化
    train_loss = loss.item()

    # 予測ラベル(1 or 0)計算
    predicted = torch.where(outputs < 0.0, 0, 1)

    # 精度計算
    train_acc = (predicted == labels1).sum() / len(y_train)

    # 予測フェーズ

    # 予測計算
    outputs_test = net(inputs_test)

    # 損失計算
    loss_test = criterion(outputs_test, labels1_test)

    # 損失のスカラー化
    val_loss =  loss_test.item()

    #予測ラベル(1 or 0)計算
    predicted_test = torch.where(outputs_test < 0.0, 0, 1)

    # 精度計算
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# %% id="-k1KOgubPsqq"
#損失と精度の確認

print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# %% id="JNkolGE1Psqr"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b', label='訓練')
plt.plot(history[:,0], history[:,3], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.legend()
plt.show()

# %% id="L6JFhi1DPsqr"
# 学習曲線の表示 (精度)

plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線(精度)')
plt.legend()
plt.show()

# %% id="FQC_gdBRPsqr"
# パラメータの取得

bias = net.l1.bias.data.numpy()
weight = net.l1.weight.data.numpy()
print(f'BIAS = {bias}, WEIGHT = {weight}')

# 決定境界描画用 x1の値から x2の値を計算する
def decision(x):
    return(-(bias + weight[0,0] * x)/ weight[0,1])

# 散布図のx1の最小値と最大値
xl = np.array([x_test[:,0].min(), x_test[:,0].max()])
yl = decision(xl)

# 結果確認
print(f'xl = {xl}  yl = {yl}')

# %% id="XfKQ7-mQPsqs"
# 散布図表示
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x',
        c='b', s=50, label='class 0')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o',
        c='k', s=50, label='class 1')

# 決定境界直線
plt.plot(xl, yl, c='b')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()

# %% id="psmSY4pc9pxw"
