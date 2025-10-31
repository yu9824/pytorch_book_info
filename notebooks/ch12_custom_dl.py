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
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/yu9824/pytorch_book_info/blob/main/ch12_custom_dl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="6LF5_Ea6qRFQ"
# # 12章 カスタムデータによる画像認識

# %% id="Rmvqz-691Wym" colab={"base_uri": "https://localhost:8080/"} outputId="0ed8d351-3c7a-44a9-8faa-b4e0fa4afd7c"
# 必要ライブラリ・コマンドの導入

# !pip install japanize_matplotlib | tail -n 1
# !pip install torchviz | tail -n 1
# !pip install torchinfo | tail -n 1
# w = !apt install tree
print(w[-2])

# %% id="EBjRX49eqRFd"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# %% id="hJy-mzVHqRFi"
# PyTorch関連ライブラリのインポート

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# %% id="zx2Zkbou1Nfc"
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

# %% id="cqa0F-rK1WZu" colab={"base_uri": "https://localhost:8080/"} outputId="75278e56-01bc-4f80-f2d3-9030dcf65a04"
# GPUチェック

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown] id="aXvDFn2a1bcc"
# ### 共通関数の読み込み

# %% id="bLJx4mjqGsj1" colab={"base_uri": "https://localhost:8080/"} outputId="37dc188d-1047-4ae2-aff3-30634096117d"
# 共通関数のダウンロード
# !git clone https://github.com/makaishi2/pythonlibs.git

# 共通関数のロード
from pythonlibs.torch_lib1 import *

# 共通関数の存在チェック
print(README)

# %% [markdown] id="obUglB5x18Lk"
# ## 12.3 データ準備

# %% [markdown] id="ZlHS2MHb2qer"
# ### サンプルデータのダウンロード・解凍

# %% id="-utQupbe6dYb" colab={"base_uri": "https://localhost:8080/"} outputId="dc212b1d-5357-4388-a3e2-c56c7d7df067"
# サンプルデータのダウンロード
# w = !wget -nc https://download.pytorch.org/tutorial/hymenoptera_data.zip

# 結果確認
print(w[-2])

# %% id="TpWlbLEiwX1e" colab={"base_uri": "https://localhost:8080/"} outputId="d549be2a-8a79-4e80-b0d1-324a75125f19"
# データ解凍
# w = !unzip -o hymenoptera_data.zip

# 結果確認
print(w[-1])

# %% id="PieDzc7gLsr_" colab={"base_uri": "https://localhost:8080/"} outputId="6ed79105-05c2-4fa8-c228-38c5a164597a"
# 解凍ファイルのtree表示
# !tree hymenoptera_data

# %% [markdown] id="kLaShsdq4ESy"
# ### Transforms定義

# %% id="ZyHMsmhyqRFu"
# Transforms定義

# 検証データ用 : 正規化のみ実施
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# 訓練データ用: 正規化に追加で反転とRandomErasingを実施
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# %% [markdown] id="pBOU0Tz44f4v"
# ### データセット定義

# %% id="4aT49QBiqRFr" colab={"base_uri": "https://localhost:8080/"} outputId="1b616436-e5ca-4366-e3c9-990afabc5bf1"
# ツリーのベースディレクトリ
data_dir = 'hymenoptera_data'

# 訓練データディレクトリと検証データディレクトリの指定
import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# join関数の結果確認
print(train_dir, test_dir)

# 分類先クラスのリスト作成
classes = ['ants', 'bees']

# %% id="04h9gxZ54jrm"
# データセット定義

# 訓練用
train_data = datasets.ImageFolder(train_dir,
            transform=train_transform)
# 訓練データのイメージ表示用
train_data2 = datasets.ImageFolder(train_dir,
            transform=test_transform)
# 検証用
test_data = datasets.ImageFolder(test_dir,
            transform=test_transform)

# %% id="xCCha5Mc52s0" colab={"base_uri": "https://localhost:8080/"} outputId="3c6c56d4-2cfc-4de4-83e4-2a90ce14533c"
# データ件数確認

print(f'訓練データ: {len(train_data)}件')
print(f'検証データ: {len(test_data)}件')

# %% id="46ha7-dFnPqx" colab={"base_uri": "https://localhost:8080/", "height": 285} outputId="7ac007d0-c775-46c1-8870-6b74d1398842"
# 検証データ
# 最初の10個と最後の10個の表示

plt.figure(figsize=(15, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    image, label = test_data[i]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 11)
    image, label = test_data[-i-1]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# %% [markdown] id="CCSHyuQl451E"
# ### データローダー定義

# %% id="A_hhZVqn4j2q"
# データローダー定義

batch_size = 10

# 訓練用
train_loader = DataLoader(train_data,
      batch_size=batch_size, shuffle=True)

# 検証用
test_loader = DataLoader(test_data,
      batch_size=batch_size, shuffle=False)

# イメージ表示用
train_loader2 = DataLoader(train_data2,
      batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data,
      batch_size=50, shuffle=True)

# %% [markdown] id="WWsJGBllaxOB"
# ### イメージ表示

# %% id="g1jc5OO-a3ay" colab={"base_uri": "https://localhost:8080/", "height": 753} outputId="1c81b4ce-b601-49fc-e4ec-c15a9b197040"
# 検証用データ(50件)
torch_seed()
show_images_labels(test_loader2, classes, None, None)

# %% [markdown] id="OEwE_MhaNFcQ"
# ## 12.4 ファインチューニング版

# %% id="qxaYEOpNqRF5" colab={"base_uri": "https://localhost:8080/"} outputId="96aa59a3-b577-451f-fb1e-ed39f1017624"
# ファインチューニング版

# 学習済みモデルの読み込み
# vgg19_bnをパラメータ付きで読み込む
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# 乱数初期化
torch_seed()

# 最終ノードの出力を2に変更する
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d関数の取り外し
net.avgpool = nn.Identity()

# GPUの利用
net = net.to(device)

# 学習率
lr = 0.001

# 損失関数定義
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

# historyファイルも同時に初期化する
history = np.zeros((0, 5))

# %% id="DoRR1aOJxwsG" colab={"base_uri": "https://localhost:8080/", "height": 266, "referenced_widgets": ["1a38814903e04d18a285b5f36d5c4d0e", "98c9f90bf75243769f5122626e7c0d1a", "53069717d6764c77a136bf3fa20b27f7", "93920013dcfe4096927b5917bab86017", "eb6ea628fe7d4e58af88feb4ce4004d5", "494255629a3440fb8a9d4cbed3489fe0", "0014b01ec2f7449e9fcee425beef72ff", "1ca138af329e40f990ccff23ff3431c8", "5a45b571ba894c19898d38fd43f82ef9", "c489cec64bcd4fa0b31243787f2bab55", "5ad345e9edcf4bc8af4741715fd677bf", "e890371f9cbd4cbba3fba4f1b7fd67bc", "15dad84ea32c45bebcf6e9bb6eee74f3", "0858ed8d64e6456e85dae29fff981d27", "b3a648b8bcd3462b8c037618415a65e3", "bd9f42f83bda4511a8edff61c4a04da0", "41ebd1c54e194174b4351a42d88146d9", "57fb820d5e164a479766263e1f302026", "156ca1213041462b9ef48012fdb917f3", "8114b7f792fa42f49234742f17f7c0d1", "295e076ac7244d9a845542a3837c1b5f", "ec20399518484d95b0e10aeb75fbded4", "f29ade515515475d9b362162ab8261b5", "2182848e3f9f4876bbee6182473c2659", "31bc844bb5c1468c9a807627314844c4", "2059aa1c7861464f9a6e56de43a40715", "be48cf743b054cf0a1e1f82f5d79b2c2", "c86885bb35084f25801ea77acbc91184", "938bd93daa124b44bfb867d494dfa5e4", "27dc5c523d7e415ea89ec2f7be28c774", "2c06a82d7ec64f06bb4e9a8804f79207", "35710a52c7a445ed8c9e9d7dc7eec29c", "609c7813bdc14db993316ab69cb49c7e", "e02d07a8bbdf4c1690f40f0e4c524bf1", "9d0ceb53684f4786ad0fccf030c5474c", "9799a30ab26a4611bf0a5320421c7dfb", "3f8b3082464741c6a4986d47592206fb", "9131ea639b9d4c86aaa566246a83d81c", "6f7eba80542c4965b4cb72824caec262", "a8ea45e0850a40bfa0ba6759de1e91f6", "b2a43fdb08ec498f901ba925a610a9ed", "aac8d67594c94b0f973ea80955efc68f", "0d3a4a7ba1d0417e8b9fdfb14746549a", "6bc2228ca47d481ab3f0ad7866f3c017", "baf2120209c846fcbca4f64c33e85386", "28ce010ea48b4808b5879b28863b2226", "fa6011b4dc8d416ba267e65ffa3adc61", "9ad4987eb0c448ad85ff1f176df2bd81", "7fbf8054488b4306b72c578a23f30d20", "31a345b76041428ab888f77be046389a", "ab2dca1cfb6c40318ec2123c53f9eefa", "0e256aaa7df44cee8e154f55cf520474", "479c31d06d294d55ac9cc8a67f844379", "62f1fd323a5e42f1b5f58137318eb50d", "2fe8f60439ae4f48ac007005a1de442b"]} outputId="8688180d-48ab-4c5e-e51b-236a266858c2"
# 学習
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
          train_loader, test_loader, device, history)

# %% id="8g21SeazNhWM" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="e9d1def4-27c5-4678-de3a-34dbce5b5d72"
# 結果確認
evaluate_history(history)

# %% id="C3cR0v8DT0C-" colab={"base_uri": "https://localhost:8080/", "height": 752} outputId="19c5c00e-c47c-472b-b763-6957db09c0c1"
# 乱数初期化
torch_seed()

# 検証データへの結果表示
show_images_labels(test_loader2, classes, net, device)

# %% [markdown] id="jsszNoIXN5NW"
#
# ## 12.5 転移学習版

# %% id="Wm5kQFABbyGm"
# vgg19_bnをパラメータ付きで読み込む
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# すべてのパラメータで勾配計算なしに
for param in net.parameters():
    param.requires_grad = False

# 乱数初期化
torch_seed()

# 最終ノードの出力を2に変更する
# このノードのみ勾配計算をすることになる
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d関数の取り外し
net.avgpool = nn.Identity()

# GPUの利用
net = net.to(device)

# 学習率
lr = 0.001

# 損失関数定義
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
# パラメータ修正の対象を最終ノードに限定
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# historyファイルも同時に初期化する
history = np.zeros((0, 5))

# %% id="lYe7az8QxF6k" colab={"base_uri": "https://localhost:8080/", "height": 266, "referenced_widgets": ["c08814ffb37a455d86a112954ff7139e", "bec65ffdf5184a969549dda6970b09f3", "7e9ae9c6381b46cd99d2a6f891abb44b", "47f27f8bc41d4c6cb2e3a4b340f0a147", "8a177a1b78e74634af73c22ecd3b7bed", "2156c3b70e46490ab07a34e8adf8f932", "b0d5bb81c61a44e7952fa129d24be57a", "4c7b3491cd0b443d8d3549f44d4189c3", "315e5c996e6342d4bfa51d840af32458", "24f0316c394a4333a2450778ba8b61bc", "de412aff136d477a9ab27705124f35da", "9e59f4692f6f4523a138e946cd9bfc10", "7fc9a90ddacb4ab9a8dcfea98a95451d", "4b4c009dc0bc40f9ac59974453086324", "034b886db13846d8bff52c7ed9a13ae1", "3334e0a5265a48c991e3740d5e1c7260", "6a883aa4d50b46aba2847a7baf2287f8", "9e023601dc484993a5d7708fc67a2682", "5cd6d51b0d654e87b7327eebda1fe3c5", "21667a0815fe4b41baf30ee63c5a1c44", "8ccb382389df4eada16e4aa7b23dd4b2", "578f3113d4c849b2b076e7bd8600d021", "2034cf634c2a4be5bff72955907beaab", "5a1e70ffcd5d450a8af0700f399c90d0", "f4851386aaf84b308a3cd8a3a8840fec", "0567ea26869440dd90a9a24e4175d4fd", "d2487128922948f0a9487e2f889f7a7e", "9b3577baea5741308af19e3d9cd2b4de", "36dc07a0d079430195d7e48e67c32e97", "5c63e7d382a44b3f8158f96bfa017267", "38f1ff39d39147e08f0625eee5d090e7", "e876178c52bb4af98380ae6c3279af41", "21e90ab16ec1413194717c63ecaa1d3e", "9a916aaf09d14e4c90eafc496a9129a6", "03052ff31dcf45fcb2e9437c0f7a8743", "5cd0328cb8594d3d999948a7f19fdf1e", "e9a1db737929449d9fcd9aeac475b8c5", "bff5f8634ac84eec854c8c9e8a3065c9", "27be4ea9161a417ab4e40f13b3be7b41", "4fd5400f829541a08e21f73e8fdcec0b", "841fa0fa8a6f4aa5b67684e0896cb3c6", "204660318b3f483bab9a69978a2bba34", "2390b058a71a4d68bac90a2bf54e35d8", "dcb257ce927b4bc49f46122801c5bfbe", "bab7dec4b1004d3e822f3fd043ed533a", "ecfe8868dad5437da859414fb34c8e6d", "1309b4de238d44f1be75843e03fa9f85", "c27e21ff797b4e38bc417b6667dd51c6", "5b56285f35764a83ac21185e87368803", "31bf96361a754cd7b2d9fee61116907d", "54914da2bc92465b92f311a354ca5320", "3de2ff56985f43a8ae2cea2096daaf79", "60cf45c2eeba44baa7d3a84dd25ca38b", "18b6372bfe8e424cad16149ca5e6ecd3", "1c5a053ac2624f5f8fa94b8e32098086"]} outputId="320e5c6b-f8ad-49ba-b841-ddab38df6ae9"
# 学習
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
          train_loader, test_loader, device, history)

# %% id="HQmAWPvmqRF9" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="69d5c3a2-fa19-459e-a0eb-e6e7d1ccd097"
# 結果サマリー
evaluate_history(history)

# %% id="YWcXW3D4OvBW" colab={"base_uri": "https://localhost:8080/", "height": 752} outputId="f77c6e0a-8646-4a82-ada6-0d2a0a99611c"
# 乱数初期化
torch_seed()

# 検証データへの結果表示
show_images_labels(test_loader2, classes, net, device)

# %% [markdown] id="hEFpvwzL3Yxn"
# ## 12.6 ユーザー定義データの場合
# シベリアンハスキーとオオカミの画像を利用  
# ダウンロード元  
# https://pixabay.com/ja/

# %% [markdown] id="Z4HwQvQ-BCqI"
# ### データダウンロード・解凍

# %% id="HS7hQ-hwLssE" colab={"base_uri": "https://localhost:8080/"} outputId="a54dd7b9-931a-4ad3-85dd-2cc88a259a8e"
# データダウンロード
# w = !wget https://github.com/makaishi2/pythonlibs/raw/main/images/dog_wolf.zip
print(w[-2])

# 解凍
# !unzip dog_wolf.zip | tail -n 1

# 解凍結果のツリー表示
# !tree dog_wolf

# %% [markdown] id="l9SqXhI2VEI4"
# ### Transforms定義

# %% id="1RLUFj7RLssE"
# Transforms定義

# 検証データ用 : 正規化のみ実施
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# 訓練データ用: 正規化に追加で反転とRandomErasingを実施
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# %% [markdown] id="gc2t3uIHdSaw"
# ### データセット定義

# %% id="YsaX1qNQ4HLf"
# データセット定義

data_dir = 'dog_wolf'

import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

classes = ['dog', 'wolf']

train_data = datasets.ImageFolder(train_dir,
            transform=train_transform)
train_data2 = datasets.ImageFolder(train_dir,
            transform=test_transform)
test_data = datasets.ImageFolder(test_dir,
            transform=test_transform)


# %% id="4sz6Hcv7B9Ln" colab={"base_uri": "https://localhost:8080/"} outputId="4ebb03e6-9850-4186-c2a5-a7ca3d085781"
# データ件数確認

print(f'学習データ: {len(train_data)}件')
print(f'検証データ: {len(test_data)}件')

# %% [markdown] id="zW67cCTEdXTd"
# ### データローダー定義

# %% id="XiihQJiBdapk"
# データローダー定義

batch_size = 5
# 学習データ
train_loader = DataLoader(train_data,
            batch_size=batch_size, shuffle=True)
# 学習データ　イメージ表示用
train_loader2 = DataLoader(train_data2,
            batch_size=40, shuffle=False)
# 検証データ
test_loader = DataLoader(test_data,
            batch_size=batch_size, shuffle=False)
# 検証データ　イメージ表示用
test_loader2 = DataLoader(test_data,
            batch_size=10, shuffle=True)

# %% [markdown] id="58xpmTt-Vkxf"
# ### イメージ表示

# %% id="Et00TY8ZVn6Q" colab={"base_uri": "https://localhost:8080/", "height": 598} outputId="40b41dde-db0a-49bb-82bd-f5a9ddb2f302"
# 訓練用データ(４0件)
show_images_labels(train_loader2, classes, None, None)

# %% id="xYH_8XQbVoEY" colab={"base_uri": "https://localhost:8080/", "height": 134} outputId="a9792f18-afb7-43cb-a9d5-033ee1250603"
# 検証用データ(10件)
torch_seed()
show_images_labels(test_loader2, classes, None, None)

# %% [markdown] id="WEnkwdHid3rC"
# ### モデル定義 (転移学習)

# %% id="8VTSEJZz7XEz"
# 学習済みモデルの読み込み
net = models.vgg19_bn(pretrained = True)

for param in net.parameters():
    param.requires_grad = False

# 乱数初期化
torch_seed()

# 最終ノードの出力を2に変更する
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d関数の取り外し
net.avgpool = nn.Identity()

# GPUの利用
net = net.to(device)

lr = 0.001
# 損失関数定義
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
# パラメータ修正の対象を最終ノードに限定
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# historyファイルも同時に初期化する
history = np.zeros((0, 5))

# %% id="Gzdy_JX8Knc-" colab={"base_uri": "https://localhost:8080/", "height": 516, "referenced_widgets": ["f9ee2df1df6a476e8206b844ac2ad95a", "4aca87120a7f4d1b8f328ecf4feecfa2", "6520353b38444183b6ca4ac746dd3a87", "69453ea7c5094f4db4af8cea35fa1e3e", "86735466f2d0431982816e878ea9599f", "92030ce47ac744b79c5a5562cbee0fa7", "c76a0be0582d4ef6b1e9180303a893ec", "621a21c20af84fba884c5f24d2298477", "546c81b41d674473b0721e50a3e3031c", "d8e1d71595f34f8b829acd1d5fd5e8de", "09051341a9f34313b49e7579ff929db6", "b801c5cacbb34d8b9fbbf463fc555f66", "4a91777006d04586958a3b962c5ac9f6", "03bce0a6aefe489fae432496c5e23b35", "ff1a6198be9e4e98ab33d9b5bf5e456a", "83189e92ae1f4bbab7cc1cb2b1308fb6", "300775e0101a4278838e145334f32afe", "1a9ee1de4286476385803ff247d01792", "c3b803d502b145bc860de14c72c77b18", "9ea1ef9f8c4a421fb7c7fb6041d0894f", "0db7b06669d64108a0813c1068da04c0", "2d51f1f6a43349c1a4c0863ca97c4dfe", "a368a006f682452388b3116bc9792cc1", "0fdd6864f0174992822d7d2b7f978dfa", "8069c593df124fe8b7a6b871254525af", "35d41fcc8170458a9b297a6e6cf76c12", "91233f5b1ded462aa5acf1ed00df8b68", "7032f7eed12940f39257740be356399a", "d8b8235edf6a47e1bdefe07e08590202", "7867ed6679d54c5d87e21d31401e0970", "01f4bce8c1e649ec9b81e2c9034bb058", "4e6b33f1bc6b4e03856924e7bbb153e1", "217cd832cf9849dfa1de946d84fb3c07", "53bfdba5a81741c8a88bb69706d6d8c9", "eb1ce9e8082c4cf6b1a17611f869e5e4", "bb6375f9421740e98ae963f30a800db1", "e01029a052e14046838b8a74e273cb30", "0601bafdd77b414da71cf989a191d8eb", "f8c2687f410f46a3b911365aa2a878ed", "78aec43c1e03428087de3fe55f7e3d48", "a20c54ff582b4fd7bd6eb1cc15838099", "cd38ac47244f464cb902152807af8bfd", "12e563951aea4acea96da03aa152713d", "86539dcb989c441d9b26efcac2ccff25", "3f33caec0d7a457eb9984ac306833ad6", "daa89bc4082548eeaecbd4210cf1d16a", "c9f61c594b2c4c5a9a13a63985962e2b", "a676ce76f8204823baaa6bf79764938b", "d4d25b270e7c4fb09657ce17f0a1fc73", "6e4f5e91dedc464488e44d657e58de90", "287a4e086c174ae594af71dc4f8a6ab8", "9e366c71f7e44a4eae8637f39d3f6f09", "f7b5f20c786046a7a49f1723808c0ce5", "a7e29a4d756b42e2802f75589fb46aa2", "85e725d4eb1a4e909c6639fe6986bb95", "3080cae375fa41d3a9fc72d428a9b9b6", "38c2c362fe2f43da8c0f1de763d5dd1e", "37c3650459024d15b60fb0ee44ede5d9", "66911fadcdb046b4935a3e1fa08f322b", "ed15aedf80a349e58d5b7b764c3f1fc3", "97fca014a7be4e5780b226313a72671e", "216c637eaaf0410c8ebec58d1c945ef3", "cd6c9e8e24154d45b1dfdf9f469edd40", "a64d99c983a4462cad80e659d5d0834a", "ccde860a20354d9793f683b193e052c6", "5167f7cb377543d4ad2a1f012ff68ce6", "cf7705d4d6944d5f9687cb0d315d233f", "d04c13d3e17046daa6618b8b068d73b0", "2390118ea362426a9cb7e1ed7297bd26", "6e98a13a292c4f768a9df4d010d287ee", "7595971ac20d4002bfcbb09198529cb6", "33eafd6119de4f42b780d182ec4a8f00", "2d1dfb48fa8e4edc95191ef12a36564b", "8b4d7581af0c43cf889ab98965653759", "699b7d4d87db41d68cf80e2185e47afd", "a70ae1858ba0464e9aac913b3070a121", "70093fe6499b44469c08bd5cb0a2b402", "32970a775b3d47d38598abe8e886c3c5", "2819aa3a61c442af8e27e72bef58465f", "3e337ae868624feea2cb10ebfed9602a", "b1759090b69d48f2ab77050d14f08d68", "fd8fa7efb81343f293dfe197f3ac26f5", "6cc6d8f820134d0bb86b38599095b0d3", "ca4226be5dc94fa897ff04eedc9bcbae", "806cd827ee574f28a69019eb2a5c5c4d", "9302dbc976d74194975ca19bb88ff4f6", "d62e758b002b46bf9ed99eeeb5ca813c", "bd84df2a02c14d4996400b136c892347", "f2ad47155b9d414d800cae6216c19dba", "a46cbc2cadd84e9888bfa3152baa40bd", "be2c3efccb014af385896302437cf305", "aa4614d7412e421da74aa28e80ac0dc3", "86a3178e274947d69ba8bbb16b11e884", "338102198b874ee4aace236722b89922", "fddb18fd93df4aae912067512b0a20df", "3301999bc8234934918b9f993e582301", "c41113ea4de740b698977377d5d5250c", "8e2e987b4f6c47e49ca3a577382159e7", "7818461568d2405892d924dd83869dab", "844b0aa73c4344acab3e81b0edd66c0b", "a600a49050ba4e4aa9497ab10a4f89ab", "2f012ba9cd214665b3837855d50c3d96", "cb942d51f55043f58258c703e3d4f0c8", "f897fd0a83d64e7fa86dec56e7bf49f6", "eb224c0a380144278e655159182e5b5f", "d1bacd0503464b159a0fd95dcfe220b3", "18e20cd697094222a1d50a43fb8ae961", "b85e96fa0f8942c1a5e07e47bfa67942", "eea1331db0bb4fe4affdac3224c4f845", "e9362968e92145c2ac25d4caffc3747e"]} outputId="c9dab859-f1ec-4de6-8edb-cbd37ff8a157"
# 学習の実行

num_epochs = 10
history = fit(net, optimizer, criterion, num_epochs,
          train_loader, test_loader, device, history)

# %% id="JbYowp9A8JCF" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="bb547a95-0852-406f-f9e6-644e388dea49"
# 結果サマリー
evaluate_history(history)

# %% id="FdHO0-IZ_EmC" colab={"base_uri": "https://localhost:8080/", "height": 134} outputId="97e39f49-ae66-4ec7-c508-1d7f2f606008"
# 予測結果表示
torch_seed()
show_images_labels(test_loader2, classes, net, device)

# %% id="16abE2RlRv6C"


