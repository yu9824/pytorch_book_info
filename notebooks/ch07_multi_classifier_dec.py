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
# <a href="https://colab.research.google.com/github/yu9824/pytorch_book_info/blob/main/notebooks/ch07_multi_classifier_dec.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="Ch7QhC-aB8wI"
# # 7章　多値分類

# %% id="ovFKq7VJB8wJ" colab={"base_uri": "https://localhost:8080/"} outputId="1dd19489-9f67-4114-ec23-fb910e75536a"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1
# !pip install torchinfo | tail -n 1

# %% id="ZuXbWMV0B8wK"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %% id="9HGKvQdjB8wK"
# torch関連ライブラリのインポート

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot

# %% id="Xktdy7SfB8wK"
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=4)

# %% [markdown] id="pmcpTYteB8wL"
# ## 7.8 データ準備

# %% [markdown] id="LXtUTacMB8wM"
# ### データ読み込み

# %% id="DL7lXe4hB8wM" colab={"base_uri": "https://localhost:8080/"} outputId="637d0543-0355-4081-d419-84bb165c7c09"
# 学習用データ準備

# ライブラリのインポート
from sklearn.datasets import load_iris

# データ読み込み
iris = load_iris()

# 入力データと正解データ取得
x_org, y_org = iris.data, iris.target

# 結果確認
print('元データ', x_org.shape, y_org.shape)

# %% [markdown] id="4dOTzVWYB8wM"
# ### データ絞り込み

# %% id="FgWVyVe_B8wM" colab={"base_uri": "https://localhost:8080/"} outputId="f60d34ae-d764-4bd1-d086-40a2c755de95"
# データ絞り込み

# 入力データに関しては、sepal length(0)とpetal length(2)のみ抽出
x_select = x_org[:,[0,2]]

# 結果確認
print('元データ', x_select.shape, y_org.shape)

# %% [markdown] id="SDu-VbBQB8wM"
# ### 訓練データ・検証データの分割

# %% id="gHtoGNaiB8wN" colab={"base_uri": "https://localhost:8080/"} outputId="4f4ac32c-e5f7-4919-9912-20c577023685"
# 訓練データ、検証データに分割 (シャフルも同時に実施)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_select, y_org, train_size=75, test_size=75,
    random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %% [markdown] id="hn8In74JB8wN"
# ### 訓練データの散布図表示

# %% id="5duf5mMpB8wN"
# データを正解値ごとに分割

x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
x_t2 = x_train[y_train == 2]

# %% id="tYaWqq7IB8wN" colab={"base_uri": "https://localhost:8080/", "height": 550} outputId="80aafdbf-d222-4fb7-fb75-ecc58acae42d"
# 散布図の表示

plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolour)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (virginica)')
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.legend()
plt.show()

# %% [markdown] id="p6jQZHH6B8wN"
# ## 7.9 モデル定義

# %% id="fUx0cencB8wO" colab={"base_uri": "https://localhost:8080/"} outputId="240c9a56-d3a7-487a-ae47-ea61b6fa0b21"
# 学習用パラメータ設定

# 入力次元数
n_input = x_train.shape[1]

# 出力次元数
# 分類先クラス数　今回は3になる
n_output = len(list(set(y_train)))

# 結果確認
print(f'n_input: {n_input}  n_output: {n_output}')


# %% id="Q5UmhWa2B8wO"
# モデルの定義
# 2入力3出力のロジスティック回帰モデル

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # 初期値を全部1にする
        # 「ディープラーニングの数学」と条件を合わせる目的
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1

# インスタンスの生成
net = Net(n_input, n_output)

# %% [markdown] id="gVNOGdyTB8wO"
# ### モデル確認

# %% id="qHvoNQRRB8wO" colab={"base_uri": "https://localhost:8080/"} outputId="f1878668-7610-49d5-92ee-20f37aaf4e9f"
# モデル内のパラメータの確認
# l1.weightが行列にl1.biasがベクトルになっている

for parameter in net.named_parameters():
    print(parameter)

# %% id="wiUZvDi_B8wP" colab={"base_uri": "https://localhost:8080/"} outputId="4f5c857d-b861-4769-e987-0e5abed92bdb"
# モデルの概要表示

print(net)

# %% id="nNobe3FvB8wP" colab={"base_uri": "https://localhost:8080/"} outputId="4b32d512-8b3d-49cc-ff49-11af5ef649a4"
# モデルのサマリー表示

summary(net, (2,))

# %% [markdown] id="-sBrtt-ZB8wP"
# ### 最適化アルゴリズムと損失関数

# %% id="LmEw7VFWB8wP"
# 損失関数： 交差エントロピー関数
criterion = nn.CrossEntropyLoss()

# 学習率
lr = 0.01

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# %% [markdown] id="508vYI0zB8wP"
# ## 7.10 勾配降下法

# %% [markdown] id="KKKMMHq2B8wP"
# ### データのテンソル化

# %% id="cOg9_jE9B8wP"
# 入力変数x_trainと正解値 y_trainのテンソル変数化

inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).long()

# 検証用変数のテンソル変数化

inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).long()

# %% [markdown] id="Bo-FPvbQB8wQ"
# ### 計算グラフの可視化

# %% id="Tv663B7hB8wQ" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="81237ee5-4526-43d9-e033-6d21995d8b12"
# 予測計算
outputs = net(inputs)

#  損失計算
loss = criterion(outputs, labels)

# 損失の計算グラフ可視化
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# %% [markdown] id="5cbHG6zdB8wQ"
# ### 予測ラベル値の取得方法

# %% id="2fQ-IYOtB8wR" colab={"base_uri": "https://localhost:8080/"} outputId="635d3ab4-5e54-40b9-b916-94f4db91483c"
# torch.max関数呼び出し
# 2つめの引数は軸を意味している。1だと行ごとの集計。
print(torch.max(outputs, 1))

# %% colab={"base_uri": "https://localhost:8080/"} id="IjBxRf9e-_cg" outputId="29476ac9-6e84-4738-ff96-c1cc0b486325"
outputs[:5] # 全部同じなため、indexとしては0が得られる。

# %% colab={"base_uri": "https://localhost:8080/"} id="oLeHfntI-RLs" outputId="24f2a6ee-d073-44bc-888a-fb4e84c21f10"
print(outputs.shape)
print(torch.max(outputs, axis=1))

# %% id="LYW8R14iB8wR" colab={"base_uri": "https://localhost:8080/"} outputId="ba116a3b-96a2-467d-ff62-80ca8b335ef0"
# ラベル値の配列を取得
torch.max(outputs, 1)[1]

# %% colab={"base_uri": "https://localhost:8080/"} id="ThMAri-c-xVO" outputId="bd756109-beab-49ac-9944-050f7eeaa604"
# ラベル値の配列を取得
torch.max(outputs, 1).indices   # こっちでもよい。こっちのほうがわかりやすい気がする。

# %% [markdown] id="nUq1owdoB8wR"
# ### 繰り返し計算

# %% id="p6vnUjx8B8wR"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： 交差エントロピー関数
criterion = nn.CrossEntropyLoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 10000

# 評価結果記録用
history = np.zeros((0,5))

# %% id="dky9vOVqB8wR" colab={"base_uri": "https://localhost:8080/"} outputId="8b2c6ff6-d337-44b6-b52f-4c5d87799d05"
# 繰り返し計算メインループ

for epoch in range(num_epochs):

    # 訓練フェーズ

    #勾配の初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 損失計算
    loss = criterion(outputs, labels)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    # 予測ラベル算出
    predicted = torch.max(outputs, 1)[1]

    # 損失と精度の計算
    train_loss = loss.item()
    train_acc = (predicted == labels).sum()  / len(labels)

    #予測フェーズ

    # 予測計算
    outputs_test = net(inputs_test)

    # 損失計算
    loss_test = criterion(outputs_test, labels_test)

    # 予測ラベル算出
    predicted_test = torch.max(outputs_test, 1)[1]

    # 損失と精度の計算
    val_loss =  loss_test.item()
    val_acc =  (predicted_test == labels_test).sum() / len(labels_test)

    if ((epoch) % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# %% [markdown] id="JOztcpv3B8wS"
# ## 7.11 結果確認

# %% id="Q5EZQ946B8wS" colab={"base_uri": "https://localhost:8080/"} outputId="df4ca3be-34d1-4182-b482-448d29f6853f"
#損失と精度の確認

print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# %% id="LbibwG3bB8wS" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="accea5a5-c931-4635-9261-5c67caa4396c"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b', label='訓練')
plt.plot(history[:,0], history[:,3], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.legend()
plt.show()

# %% id="WJnWmovUB8wS" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="6ef85df0-c172-4612-81bd-896c2fbd6070"
# 学習曲線の表示 (精度)

plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線(精度)')
plt.legend()
plt.show()

# %% [markdown] id="JG3V10NNB8wT"
# ### モデルへの入力と出力の確認

# %% id="vWjT02ChB8wT" colab={"base_uri": "https://localhost:8080/"} outputId="79433559-a5b9-4b52-8992-e9cae1a07056"
# 正解データの0番目、2番目、3番目

print(labels[[0,2,3]])

# %% id="2j8v5mnCB8wT" colab={"base_uri": "https://localhost:8080/"} outputId="dfccc556-54f5-4053-8a37-0c6726241bea"
# 該当する入力値を抽出

i3 = inputs[[0,2,3],:]
print(i3.data.numpy())

# %% id="9skIo4OwB8wT" colab={"base_uri": "https://localhost:8080/"} outputId="367a42c8-03a7-4126-ba48-24734d2abec7"
# 出力値にsoftmax関数をかけた結果を取得

softmax = torch.nn.Softmax(dim=1)
o3 = net(i3)
k3 = softmax(o3)
print(o3.data.numpy())
print(k3.data.numpy())

# %% [markdown] id="uPlEVs83B8wT"
# ### 最終的な重み行列とバイアスの値

# %% id="rA6qeTWHB8wU" colab={"base_uri": "https://localhost:8080/"} outputId="aa8070ce-a60c-498b-8059-659dc3f6aacd"
# 重み行列
print(net.l1.weight.data)

# バイアス
print(net.l1.bias.data)

# %% [markdown] id="IdIvK-6E7-BM"
# ## 決定境界の描画

# %% [markdown] id="sd3_pOqI7-BM"
# ### 描画領域計算

# %% colab={"base_uri": "https://localhost:8080/"} id="2Jg6newF7-BM" outputId="4bab6e2b-b702-4f47-cfac-0814ded36d04"
# x, yの描画領域計算
x_min = x_train[:,0].min()
x_max = x_train[:,0].max()
y_min = x_train[:,1].min()
y_max = x_train[:,1].max()
x_bound = torch.tensor([x_min, x_max])

# 結果確認
print(x_bound)


# %% id="trfhb2Bo7-BM"
# 決定境界用の１次関数定義
def d_bound(x, i, W, B):
    W1 = W[[2,0,1],:]
    W2 = W - W1
    w = W2[i,:]
    B1 = B[[2,0,1]]
    B2 = B - B1
    b = B2[i]
    v = -1/w[1]*(w[0]*x + b)
    return v


# %% colab={"base_uri": "https://localhost:8080/"} id="NC5sIacJ7-BM" outputId="63835f3d-ef35-45c6-fcb0-438b453e824b"
# 決定境界のyの値を計算

W = net.l1.weight.data
B = net.l1.bias.data

y0_bound = d_bound(x_bound, 0, W, B)
y1_bound = d_bound(x_bound, 1, W, B)
y2_bound = d_bound(x_bound, 2, W, B)

# 結果確認
print(y0_bound)
print(y1_bound)
print(y2_bound)

# %% colab={"base_uri": "https://localhost:8080/", "height": 550} id="e8Smy9SH7-BM" outputId="36871eb9-fc00-474a-af91-79cc7e6589f1"
# 散布図と決定境界の標示

# xとyの範囲を明示的に指定
plt.axis([x_min, x_max, y_min, y_max])

# 散布図
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolour)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (virginica)')

# 決定境界
plt.plot(x_bound, y0_bound, label='2_0')
plt.plot(x_bound, y1_bound, linestyle=':',label='0_1')
plt.plot(x_bound, y2_bound,linestyle='-.',label='1_2')

# 軸ラベルと凡例
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.legend()
plt.show()

# %% [markdown] id="m016TNL9B8wU"
# ## 7.12 入力変数の4次元化

# %% id="HI61G1-GB8wU" colab={"base_uri": "https://localhost:8080/"} outputId="512fccbc-39f7-43f6-8c0c-b828c2df86d4"
# 訓練データ、検証データに分割 (シャフルも同時に実施)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_org, y_org, train_size=75, test_size=75,
    random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 入力次元数
n_input = x_train.shape[1]

# %% id="ziivFupuB8wU" colab={"base_uri": "https://localhost:8080/"} outputId="be046c88-0fcb-48aa-dc6b-b64363eb3591"
print('入力データ(x)')
print(x_train[:5,:])
print(f'入力次元数: {n_input}')

# %% id="SFZ-B_yyB8wU"
# 入力データ x_train と正解データ y_train のテンソル変数化
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).long()

# 検証用データのテンソル変数化
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).long()

# %% id="DORSG_Q2B8wU"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： 交差エントロピー関数
criterion = nn.CrossEntropyLoss()

# 最適化アルゴリズム: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 10000

# 評価結果記録用
history = np.zeros((0,5))

# %% id="Y4gM9OQgB8wV" colab={"base_uri": "https://localhost:8080/"} outputId="24ca8b23-e898-4321-edb2-37b1d9696be9"
for epoch in range(num_epochs):

    # 訓練フェーズ

    #勾配の初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 損失計算
    loss = criterion(outputs, labels)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    #予測値算出
    predicted = torch.max(outputs, 1)[1]

    # 損失と精度の計算
    train_loss = loss.item()
    train_acc = (predicted == labels).sum()  / len(labels)

    #予測フェーズ

    # 予測計算
    outputs_test = net(inputs_test)

    # 損失計算
    loss_test = criterion(outputs_test, labels_test)

    # 予測ラベル算出
    predicted_test = torch.max(outputs_test, 1)[1]

    # 損失と精度の計算
    val_loss =  loss_test.item()
    val_acc =  (predicted_test == labels_test).sum() / len(labels_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch , train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# %% id="EGnDSJReB8wV" colab={"base_uri": "https://localhost:8080/"} outputId="d3fd9ff1-561a-49b5-93ca-7648acfa8f32"
# 損失と精度の確認

print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# %% id="z4NJkJDJB8wV" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="6cd57edb-e9d9-4d65-cc1d-537b1848916f"
# 学習曲線の表示 (損失)

plt.plot(history[:,0], history[:,1], 'b', label='訓練')
plt.plot(history[:,0], history[:,3], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.legend()
plt.show()

# %% id="jZPEkfLJB8wW" colab={"base_uri": "https://localhost:8080/", "height": 578} outputId="7fab233b-85df-4bb9-ee3d-b655cdf06bb8"
# 学習曲線の表示 (精度)

plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線(精度)')
plt.legend()
plt.show()

# %% [markdown] id="BwXP0yeKB8wK"
# ## コラム NLLLoss損失関数

# %% id="Lg7-oAJ4B8wL" colab={"base_uri": "https://localhost:8080/"} outputId="b21faf2a-2977-414c-f635-b6c1eab797de"
# 入力変数の準備

# 擬似的な出力データ
outputs_np = np.array(range(1, 13)).reshape((4,3))
# 擬似的な正解データ
labels_np = np.array([0, 1, 2, 0])

# Tensor化
outputs_dummy = torch.tensor(outputs_np).float()
labels_dummy = torch.tensor(labels_np).long()

# 結果確認
print(outputs_dummy.data)
print(labels_dummy.data)

# %% id="mf68l_reB8wL" colab={"base_uri": "https://localhost:8080/"} outputId="b1ca76c6-1f98-462f-db39-92bc0b456ad4"
# NLLLoss関数の呼び出し

nllloss = nn.NLLLoss()
loss = nllloss(outputs_dummy, labels_dummy)
print(loss.item())


# %% [markdown] id="QyX9w-MPB8wW"
# ## コラム 多値分類モデルの他の実装パターン

# %% [markdown] id="dYdZPF06B8wW"
# ### パターン2 モデルクラス側にLogS1oftmax関数を含める

# %% id="Cgth7H98B8wW"
# モデルの定義
# 2入力3出力のロジスティック回帰モデル

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        # softmax関数の定義
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # 初期値を全部1にする
        # 「ディープラーニングの数学」と条件を合わせる目的
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.logsoftmax(x1)
        return x2


# %% id="ZuL4yILWGMzZ"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： NLLLoss関数
criterion = nn.NLLLoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# %% id="3uyUhRWLFjt5" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="68e9beab-db34-4667-9480-5f4a9dd45ae6"
# 予測計算
outputs = net(inputs)

#  損失計算
loss = criterion(outputs, labels)

# 損失の計算グラフ可視化
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# %% id="pU9rqn5NB8wW"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： NLLLoss関数
criterion = nn.NLLLoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 10000

# 評価結果記録用
history = np.zeros((0,5))

# %% id="BsTrCuQ_B8wW" colab={"base_uri": "https://localhost:8080/"} outputId="cf17c1cf-e5a6-47ec-aed6-13c089663682"
for epoch in range(num_epochs):

    # 訓練フェーズ

    #勾配の初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # 損失計算
    loss = criterion(outputs, labels)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    #予測ラベル算出
    predicted = torch.max(outputs, 1)[1]

    # 損失と精度の計算
    train_loss = loss.item()
    train_acc = (predicted == labels).sum()  / len(labels)

    #予測フェーズ

    # 予測計算
    outputs_test = net(inputs_test)

    # 損失計算
    loss_test = criterion(outputs_test, labels_test)

    #予測ラベル算出
    predicted_test = torch.max(outputs_test, 1)[1]

    # 損失と精度の計算
    val_loss =  loss_test.item()
    val_acc =  (predicted_test == labels_test).sum() / len(labels_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch , train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# %% id="6hOWu3ETB8wX" colab={"base_uri": "https://localhost:8080/"} outputId="dca68827-4fbb-4d52-b285-9d5efdc4bb0a"
#損失と精度の確認

print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# %% id="GA3h1wFsB8wX" colab={"base_uri": "https://localhost:8080/"} outputId="5622c704-9fa4-4fef-f663-0f724c768a8c"
# パターン1モデルの出力結果
w = outputs[:5,:].data
print(w.numpy())

# 確率値を得たい場合
print(torch.exp(w).numpy())


# %% [markdown] id="eiDG6LXeB8wX"
# ### パターン3 モデルクラス側は素のsoftmax

# %% id="ZK5H4lElB8wX"
# モデルの定義
# 2入力3出力のロジスティック回帰モデル

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        # softmax関数の定義
        self.softmax = nn.Softmax(dim=1)

        # 初期値を全部1にする
        # 「ディープラーニングの数学」と条件を合わせる目的
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.softmax(x1)
        return x2


# %% id="pw5Z4sbcB8wX"
# 学習率
lr = 0.01

# 初期化
net = Net(n_input, n_output)

# 損失関数： NLLLoss関数
criterion = nn.NLLLoss()

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 10000

# 評価結果記録用
history = np.zeros((0,5))

# %% id="3qAnByPtB8wY" colab={"base_uri": "https://localhost:8080/"} outputId="a6ca02a8-2bfb-40dd-e280-5977c7c6055f"
for epoch in range(num_epochs):

    # 訓練フェーズ

    #勾配の初期化
    optimizer.zero_grad()

    # 予測計算
    outputs = net(inputs)

    # ここで対数関数にかける
    outputs2 = torch.log(outputs)

    # 損失計算
    loss = criterion(outputs2, labels)

    # 勾配計算
    loss.backward()

    # パラメータ修正
    optimizer.step()

    #予測ラベル算出
    predicted = torch.max(outputs, 1)[1]

    # 損失と精度の計算
    train_loss = loss.item()
    train_acc = (predicted == labels).sum()  / len(labels)

    #予測フェーズ

    # 予測計算
    outputs_test = net(inputs_test)

    # ここで対数関数にかける
    outputs2_test = torch.log(outputs_test)

    # 損失計算
    loss_test = criterion(outputs2_test, labels_test)

    #予測ラベル算出
    predicted_test = torch.max(outputs_test, 1)[1]

    # 対する損失と精度の計算
    val_loss =  loss_test.item()
    val_acc =  (predicted_test == labels_test).sum() / len(labels_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch , train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# %% id="Nvqi0lCOB8wY" colab={"base_uri": "https://localhost:8080/"} outputId="75e56949-18bf-461a-a323-729ec229fa42"
#損失と精度の確認

print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# %% id="W_faCAG1B8wY" colab={"base_uri": "https://localhost:8080/"} outputId="8c577c0b-dfa9-4fe1-8d64-49f9eef00b9b"
# パターン2のモデル出力値
w = outputs[:5,:].data.numpy()
print(w)

# %% id="HJ2AicieB8wY"
