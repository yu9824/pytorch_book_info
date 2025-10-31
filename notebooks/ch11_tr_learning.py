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

# %% [markdown] id="DCJd8UAnpHOP"
# # 11章 事前学習済みモデルの利用

# %% id="MDpbF1y0owR_"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1
# !pip install torchinfo | tail -n 1

# %% id="LoGUMbWTpMpJ"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %% id="3f9JdhISpSna"
# PyTorch関連ライブラリのインポート

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# %% id="QQCGCQiNpWsx"
# warning表示off
import warnings
warnings.simplefilter('ignore')

# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

# %% id="XfimLfk7pdZr"
# GPUチェック

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown] id="lkjcaoGgpxi1"
# ### 共通関数の読み込み

# %% id="EJPpA4BR37-E"
# 共通関数のダウンロード
# !git clone https://github.com/makaishi2/pythonlibs.git

# 共通関数のロード
from pythonlibs.torch_lib1 import *

# 共通関数の存在チェック
print(README)

# %% [markdown] id="gWbACFT0VCou"
# ## 11.4 nn.AdaptiveAvgPool2d関数

# %% id="xQG4sadKVCou"
# nn.AdaptiveAvgPool2dの定義
p = nn.AdaptiveAvgPool2d((1,1))
print(p)

# 線形関数の定義
l1 = nn.Linear(32, 10)
print(l1)

# %% id="d_1eVX0dVCou"
# 事前学習済みモデルのシミュレーション
inputs = torch.randn(100, 32, 16, 16)
m1 = p(inputs)
m2 = m1.view(m1.shape[0],-1)
m3 = l1(m2)

# shape確認
print(m1.shape)
print(m2.shape)
print(m3.shape)

# %% [markdown] id="LYYWn2WuqYG4"
# ## 11.5 データ準備

# %% id="u9nHcJ_PrGI6"
# 分類先クラス名の定義

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 分類先クラス数　今回は10になる
n_output = len(classes)

# %% id="9HkL6jEYqHx3"
# Transformsの定義

# 学習データ用: 正規化に追加で反転とRandomErasingを実施
transform_train = transforms.Compose([
  transforms.Resize(112),
  transforms.RandomHorizontalFlip(p=0.5), 
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5), 
  transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# 検証データ用 : 正規化のみ実施
transform = transforms.Compose([
  transforms.Resize(112),
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5)
])

# %% id="zkU-6Tkzq402"
# データ取得用関数 Dataset

data_root = './data'

train_set = datasets.CIFAR10(
    root = data_root, train = True,
    download = True, transform = transform_train)

# 検証データの取得
test_set = datasets.CIFAR10(
    root = data_root, train = False, 
    download = True, transform = transform)

# %% id="m_Mfp8iaq_gB"
# バッチサイズ指定
batch_size = 50

# データローダー

# 訓練用データローダー
# 訓練用なので、シャッフルをかける
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 検証用データローダー
# 検証時にシャッフルは不要
test_loader = DataLoader(test_set,  batch_size=batch_size, shuffle=False) 

# %% [markdown] id="OVVH4YSesPYG"
# ## 11.6 ResNet18の読み込み

# %% [markdown] id="0v7aEYoifiUv"
# ### モデルの読み込み

# %% id="wXbtwxLT0kND"
#  必要ライブラリのロード
from torchvision import models

# 事前学習済みモデルのロード
# pretraind = True で学習済みパラメータも一緒に読み込む
net = models.resnet18(pretrained = True)

# %% [markdown] id="4_fO9_oqdfCb"
# ### モデル構造の確認

# %% id="LjDl60sxCF2v"
# ネットワークの概要表示

print(net)

# %% id="bZcZ_kwvSodr"
# モデルのサマリー表示
net = net.to(device)
summary(net,(100,3,112,112))

# %% id="z3Ft9HfwPRSO"
print(net.fc)
print(net.fc.in_features)

# %% [markdown] id="BdbDrHrYCYUn"
# 最終レイヤー関数の変数名は``fc``であることがわかる

# %% [markdown] id="wCBd8M1feHZw"
# ## 11.7 最終レイヤー関数の付け替え

# %% id="WiF1qexgfwXp"
# 乱数の初期化
torch_seed()

# 最終レイヤー関数の入力次元数を確認
fc_in_features = net.fc.in_features

# 最終レイヤー関数の付け替え
net.fc = nn.Linear(fc_in_features, n_output)

# %% id="-JKSPiU86yPC"
# 確認
print(net)

# %% id="LTdygZ-o6yXa"
net = net.to(device)
summary(net,(100,3,224,224))

# %% id="Fs104unTgNjD"
# 損失の計算グラフ可視化

criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# %% id="yHAne47XVCow"
# 確認
print(net)

# %% id="CWnIZlefepwp"
# モデルのサマリー表示
net = net.to(device)
summary(net,(100,3,112,112))

# %% [markdown] id="qd5VLBVZVCox"
# ## 11.8 学習と結果評価

# %% [markdown] id="x5A8q-7QgSqH"
# ### 最適化関数などの定義

# %% id="H4WUgpRHtHPT"
# 乱数の初期化
torch_seed()

# 事前学習済みモデルのロード
# pretraind = True で学習済みパラメータも一緒に読み込む
net = models.resnet18(pretrained = True)

# 最終レイヤー関数の入力次元数を確認
fc_in_features = net.fc.in_features

# 最終レイヤー関数の付け替え
net.fc = nn.Linear(fc_in_features, n_output)

# GPUの利用
net = net.to(device)

# 学習率
lr = 0.001

# 損失関数定義
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# historyファイル初期化する
history = np.zeros((0, 5))

# %% [markdown] id="p_ew9lcof5TQ"
# ### 学習

# %% id="VCv_df0dyF0k"
# 学習
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)

# %% [markdown] id="_UXqn-JA4YrS"
# ### 結果確認

# %% id="uZ4JmNLoTGl1"
# 結果サマリー
evaluate_history(history)

# %% id="hg5Nj1Q89r2E"
# イメージと正解・予測結果の表示
show_images_labels(test_loader, classes, net, device)

# %% [markdown] id="UhYJ3MZBibla"
# ## 11.9 VGG-19-BNの利用

# %% [markdown] id="Rk7FnagBitoX"
# ###  モデルの読み込み

# %% id="h5eJTF6kiiLG"
# 事前学習済みモデルの読み込み
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# %% [markdown] id="B-tW5Li8jlK8"
# ###  モデル構造の確認

# %% id="ky1FQADwiiR4"
# モデルの確認
print(net)

# %% [markdown] id="Dw71I9bWjW9Q"
#  最終レイヤー関数は``classifier[6]``であることがわかる

# %% id="wHq_026ijIFg"
# 最終レイヤー関数の確認
print(net.classifier[6])

# %% [markdown] id="uYRQyckBkAJA"
# ### 最終レイヤー関数の付け替え

# %% id="SypKMObevaWJ"
# 乱数の初期化
torch_seed()

# 最終レイヤー関数の付け替え
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# features最後のMaxPool2dをはずす
net.features = net.features[:-1]

# AdaptiveAvgPool2dをはずす
net.avgpool = nn.Identity()

# %% id="gSxeSTSxs-IU"
# モデルのサマリー表示
net = net.to(device)
summary(net,(100,3,112,112))

# %% id="RFTQCXzxf1Z5"
# 損失の計算グラフ可視化

criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# %% [markdown] id="7pyG4IblkTd9"
# ### 最適化関数などの定義

# %% id="CdexksrdiiV4"
# 乱数の初期化
torch_seed()

# 事前学習済みモデルの読み込み
net = models.vgg19_bn(pretrained = True)

# 最終レイヤー関数の付け替え
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# features最後のMaxPool2dをはずす
net.features = net.features[:-1]

# AdaptiveAvgPool2dをはずす
net.avgpool = nn.Identity()

# モデルをGPUに送付
net = net.to(device)

# 学習率
lr = 0.001

# 損失関数定義
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# historyの初期化
history = np.zeros((0, 5))


# %% [markdown] id="493nQooHk5cj"
# ### 学習

# %% id="7AmPemwhVeGW"
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)

# %% [markdown] id="VKLNtf3TlHFr"
# ### 結果確認

# %% id="_pbeHENwlJ5E"
# 結果サマリー
evaluate_history(history)

# %% id="_47alfXWlKEO"
# イメージと正解・予測結果の表示
show_images_labels(test_loader, classes, net, device)

# %% [markdown] id="2khLcqNOYVIR"
# ## コラム CIFAR-10に転移学習を適用した場合
#

# %% id="5HCKyvj3YrHv"
# 転移学習バージョン

# 事前学習済みモデルの読み込み
net = models.resnet18(pretrained = True)

# すべてのパラメータで勾配計算をOFFにする
for param in net.parameters():
    param.requires_grad = False

# 乱数初期化
torch_seed()

# 最終レイヤー関数の付け替え
net.fc = nn.Linear(net.fc.in_features, n_output)

# GPUの利用
net = net.to(device)

# 学習率
lr = 0.001

# 損失関数定義
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
# パラメータ変更は最終レイヤー関数に限定
optimizer = optim.SGD(net.fc.parameters(), lr=lr, momentum=0.9)

# historyファイル初期化する
history = np.zeros((0, 5))

# %% id="c5XMUp3MYrKy"
# 学習

num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)

# %% id="SGoTdu5dYrPR"
# 結果サマリー

evaluate_history(history)

# %% [markdown] id="wPwr5N4dSfWE"
# ## コラム VGG-19-BNモデル修正の詳細

# %% [markdown] id="OvtXB98XTlM7"
# ### モデルの読み込み

# %% id="ItNhoRBmfKor"
# 事前学習済みモデルの読み込み
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# %% [markdown] id="tMtX3CnaTzio"
# ### モデルの構造の確認

# %% id="wdA6o1eIToDn"
print(net)

# %% [markdown] id="HXMyEQ9rUENc"
# ### 中間テンソルの確認

# %% id="z_ZNR3IxT44L"
# オリジナルデータサイズの場合
net = net.to(device)
summary(net,(100,3,224,224))

# %% id="vk4wdy77USfk"
# 実習用データサイズの場合
summary(net,(100,3,112,112))

# %% [markdown] id="e85da4waWcBd"
# ### レイヤー関数の付け替え

# %% id="9nI70QfzWf58"
# 乱数初期化
torch_seed()

# 最終レイヤー関数の付け替え
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# %% id="juNRE0nEU0eh"
# featuresの最後の要素(MaxPool2d)を落とす
net.features = net.features[:-1]
print(net.features)

# %% id="Beu83e6zVSj8"
# avgpoolに入っているAdaptiveAvgPool2dを何もしない関数(nn.Identity)に置き換え
net.avgpool = nn.Identity()

# %% [markdown] id="ml6mgUa1Wltw"
# ### 最終結果確認

# %% id="KXR3lzkDWQdl"
print(net)

# %% id="UKHU3pGgXAZ3"
# 実習用データサイズで中間テンソル確認
net = net.to(device)
summary(net,(100,3,112,112))

# %% id="I__UBcqxXKt0"
