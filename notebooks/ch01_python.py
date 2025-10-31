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
# <a href="https://colab.research.google.com/github/yu9824/pytorch_book_info/blob/main/notebooks/ch01_python.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="yoLOS2ohPSo-"
# # 1章 Python入門
# PyTorchを使ったディープラーニング・プログラミングで重要になる概念だけを抜き出して説明する

# %% id="eg2s3K8OPSpD" colab={"base_uri": "https://localhost:8080/"} outputId="d79555a9-6de2-4515-b3e7-fd0db203a1d2"
# 必要ライブラリの導入

# !pip install japanize_matplotlib | tail -n 1

# %% id="zR8TJdQQPSpD"
# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# %% id="ysxb8LzbPSpD"
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

# %% [markdown] id="zvXRG-ruPSpE"
# ## 1.2 コンテナ変数にご用心
# Pythonでは、変数は単に実際のデータ構造へのポインタに過ぎない。  
# Numpy配列などでは、このことを意識しないと思わぬ結果を招く場合がある。

# %% [markdown] id="UvPbw0jYPSpE"
# ### NumPy変数間

# %% id="mE8yGhNqPSpE" colab={"base_uri": "https://localhost:8080/"} outputId="b453835c-1d5e-4ab6-f4c6-eaec9a74f1a1"
# Numpy配列 x1 を定義
x = np.array([5, 7, 9])

# 変数yにxを代入する
# このとき、実体は共通なまま
y = x

# 結果確認
print(x)
print(y)

# %% id="hdmNhQOZPSpE" colab={"base_uri": "https://localhost:8080/"} outputId="bee9adce-2ac4-4452-8c7c-1da231c645dd"
# ここでxの特定の要素の値を変更する
x[1] = -1

# すると、yも連動して値が変わる
print(x)
print(y)

# %% id="l9H3CjJtPSpE" colab={"base_uri": "https://localhost:8080/"} outputId="1f07bf3e-9b38-4463-984b-8308ae6d2101"
# yも同時に変化して困る場合は、代入時にcopy関数を利用する
x = np.array([5, 7, 9])
y = x.copy()

# すると、xの特定の要素値の変更がyに影響しなくなる
x[1] = -1
print(x)
print(y)

# %% [markdown] id="6Xm1XIpBPSpE"
# ### テンソルとNumPy間

# %% id="FPkiYCioPSpF" colab={"base_uri": "https://localhost:8080/"} outputId="7859e6cb-d531-4bc0-85e3-7bf010a549d0"
import torch

# x1: shape=[5] となるすべて値が1テンソル
x1 = torch.ones(5)

# 結果確認
print(x1)

# x2 x1から生成したNumPy
x2 = x1.data.numpy()

# 結果確認
print(x2)

# %% id="E-SoSD6qPSpF" colab={"base_uri": "https://localhost:8080/"} outputId="e48e16d1-ea82-45ea-b5f5-4d067da464e5"
# x1の値を変更
x1[1] = -1

# 連動してx2の値も変わる
print(x1)
print(x2)

# %% id="fLRAG-SLPSpF" colab={"base_uri": "https://localhost:8080/"} outputId="b12f9846-5db8-4856-d5e2-a73358afae4c"
# 安全な方法

# x1 テンソル
x1 = torch.ones(5)

# x2 x1から生成したNumPy
x2 = x1.data.numpy().copy()

x1[1] = -1

# 結果確認
print(x1)
print(x2)


# %% [markdown] id="jAwsog3qPSpF"
# ## 1.3 数学上の合成関数とPythonの合成関数
# 数学上の合成関数がPythonでどう実装されるか確認する

# %% [markdown] id="LYchZiipPSpF"
# $f(x) = 2x^2 + 2$を関数として定義する

# %% id="3CqF27_GPSpF"
def f(x):
    return (2 * x**2 + 2)


# %% id="OiJN2YuVPSpF" colab={"base_uri": "https://localhost:8080/"} outputId="8eceb7c6-db10-4a15-a43e-8f039af993c5"
# xをnumpy配列で定義
x = np.arange(-2, 2.1, 0.25)
print(x)

# %% id="WXTDLngzPSpG" colab={"base_uri": "https://localhost:8080/"} outputId="a7dac943-d940-4dcf-9f53-cfdf3e987090"
# f(x)の結果をyに代入
y = f(x)
print(y)

# %% id="XzsseR8QPSpG" colab={"base_uri": "https://localhost:8080/", "height": 526} outputId="503db84b-6789-4306-d89f-503e7994fd1a"
# 関数のグラフ表示

plt.plot(x, y)
plt.show()


# %% id="hwdDvlSKUa8W"
# 3つの基本関数の定義
def f1(x):
    return(x**2)

def f2(x):
    return(x*2)

def f3(x):
    return(x+2)

# 合成関数を作る
x1 = f1(x)
x2 = f2(x1)
y = f3(x2)

# %% id="fNUqcDqVUbGW" colab={"base_uri": "https://localhost:8080/"} outputId="72056532-f4d3-45cb-e8bc-791bb38ba8aa"
# 合成関数の値の確認
print(y)

# %% id="hqVCcelaUbKq" colab={"base_uri": "https://localhost:8080/", "height": 526} outputId="cbe12a24-a23d-4e6f-da65-285b36231be4"
# 合成関数のグラフ表示

plt.plot(x, y)
plt.show()


# %% [markdown] id="NLsU5j_CTqRd"
# ## 1.4 数学上の微分とPythonでの数値微分実装
# Pythonでは、関数もまた、変数名は単なるポインタで、実体は別にある。  
# このことを利用すると、「関数を引数とする関数」を作ることが可能になる。

# %% [markdown] id="PEy0MMhvPSpG"
# ここで関数を数値微分する関数``diff``を定義する。  
# 数値微分の計算には、普通の微分の定義式よりいい近似式である $f'(x) = \dfrac{f(x+h)-f(x-h)}{2h}$を利用する。

# %% [markdown] id="CkJkTyHAtT0U"
# 上記は「中心差分」と呼ぶ。一方、教科書で最初に習う微分の定義式は前方差分と呼ぶ。
#
# 中心差分の方が誤差が少ないことはテイラー展開を用いて証明が可能。
#
# - 参考: https://mochablog.org/diff-forward-central/

# %% id="DUk6vPjvlWgc"
# 関数を微分する関数fdiffの定義
def fdiff(f):
    # 関数fを引数に微分した結果の関数をdiffとして定義
    def diff(x):
        h = 1e-6
        return (f(x+h) - f(x-h)) / (2*h)

    # fdiffの戻りは微分した結果の関数diff
    return diff


# %% [markdown] id="AwinTXaKrEpU"
# 2次関数fに対して、今作った関数fdiffを適用して、数値微分計算をしてみる。

# %% id="CFUT1Mmslf4n" colab={"base_uri": "https://localhost:8080/"} outputId="8dc3505d-0e85-49b5-cd2b-481d4af04750"
# 2次関数の数値微分

# fの微分結果の関数diffを取得
diff = fdiff(f)

# 微分結果を計算しy_dashに代入
y_dash = diff(x)

# 結果確認
print(y_dash)

# %% id="vQsqq3xlPSpG" colab={"base_uri": "https://localhost:8080/", "height": 526} outputId="9234bd71-2acb-46ea-b1dd-7628273b7951"
# 結果のグラフ表示
plt.plot(x, y, label=r'y = f(x)', c='b')
plt.plot(x, y_dash, label=r"y = f '(x)", c='k')
plt.legend()
plt.show()


# %% [markdown] id="L22xQBOAPSpH"
# シグモイド関数 $g(x) = \dfrac{1}{1 + \exp(-x)}$に対して同じことをやってみる。

# %% id="KU6ZSfbEPSpH"
# シグモイド関数の定義
def g(x):
    return 1 / (1 + np.exp(-x))


# %% id="P4_TAZkiPSpH" colab={"base_uri": "https://localhost:8080/"} outputId="8166f9b6-24b6-4d7e-8a62-e8deb5c5bb68"
# シグモイド関数の計算
y = g(x)
print(y)

# %% id="KxnzGh76PSpH" colab={"base_uri": "https://localhost:8080/", "height": 526} outputId="186e314f-a2f9-4742-b0ca-a3c8caffb71f"
# 関数のグラフ表示

plt.plot(x, y)
plt.show()

# %% id="TDlA_t5cPSpH" colab={"base_uri": "https://localhost:8080/"} outputId="a79b68aa-c8c8-47ce-e136-6e069e27dfb7"
# シグモイド関数の数値微分

# gを微分した関数を取得
diff = fdiff(g)

# diffを用いて微分結果y_dashを計算
y_dash = diff(x)

# 結果確認
print(y_dash)

# %% id="Iu_cT2v_PSpH" colab={"base_uri": "https://localhost:8080/", "height": 526} outputId="b28bf8d0-8c21-4896-d3e3-06e9d9dc494f"
# 結果のグラフ表示
plt.plot(x, y, label=r'y = f(x)', c='b')
plt.plot(x, y_dash, label=r"y = f '(x)", c='k')
plt.legend()
plt.show()

# %% [markdown] id="U6uqQ6L4PSpH"
# シグモイド関数の微分結果は$y(1-y)$となることがわかっている。  
# これはyの二次関数で、$y=\dfrac{1}{2}$の時に最大値$\dfrac{1}{4}$を取る。  
# 上のグラフはその結果と一致していて、数値微分が正しくできていることがわかる。

# %% [markdown] id="Jp1YdzDvPSpH"
# ## 1.5 オブジェクト指向プログラミング入門

# %% id="Ab0LDmO6PSpH"
# グラフ描画用ライブラリ
import matplotlib.pyplot as plt

# 円描画に必要なライブラリ
import matplotlib.patches as patches


# %% id="ipJlePaSPSpI"
# クラス Point の定義

class Point:
    # インスタンス生成時にxとyの２つの引数を持つ
    def __init__(self, x, y):
        # インスタンスの属性xに第一引数をセットする
        self.x = x
        # インスタンスの属性yに第二引数をセットする
        self.y = y
    # 描画関数 drawの定義 (引数はなし)
    def draw(self):
        # (x, y)に点を描画する
        plt.plot(self.x, self.y, marker='o', markersize=10, c='k')


# %% id="DAtC1ASWPSpI"
# クラスPointからインスタンス変数p1とp2を生成する
p1 = Point(2,3)
p2 = Point(-1, -2)

# %% id="nuPBuQQyPSpI" colab={"base_uri": "https://localhost:8080/"} outputId="0d211415-38f9-4625-8585-335600f48c86"
# p1とp2の属性x, yの参照
print(p1.x, p1.y)
print(p2.x, p2.y)

# %% id="P1RcK4uhPSpI" colab={"base_uri": "https://localhost:8080/", "height": 533} outputId="4a723153-6b14-40ea-d553-a742f2dc9be1"
# p1とp2のdraw関数を呼び出し、2つの点を描画する
p1.draw()
p2.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


# %% id="EuESwqVcPSpI"
# Pointの子クラスCircleの定義その1
class Circle1(Point):
    # Circleはインスタンス生成時に引数x,y,rを持つ
    def __init__(self, x, y, r):
        # xとyは、親クラスの属性として設定
        super().__init__(x, y)
        # rは、Circleの属性として設定
        self.r = r

    # この段階でdraw関数は定義しない


# %% id="eIDjNrTaPSpI"
# クラスCircleからインスタンス変数c1_1を生成する
c1_1 = Circle1(1, 0, 2)

# %% id="BEHIwmDzPSpI" colab={"base_uri": "https://localhost:8080/"} outputId="fd992a52-b1d9-4bd8-c190-ddf92dea1e7c"
# c1_1の属性の確認
print(c1_1.x, c1_1.y, c1_1.r)

# %% id="BLkCbcCiPSpI" colab={"base_uri": "https://localhost:8080/", "height": 533} outputId="bc07c997-4d06-4549-9d7b-4797309c2702"
# p1, p2, c1_1 のそれぞれのfraw関数を呼び出す
ax = plt.subplot()
p1.draw()
p2.draw()
c1_1.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


# %% [markdown] id="obg1_EqDPSpJ"
# この段階でdraw関数は親で定義した関数が呼ばれていることがわかる

# %% id="5qXE1-RCPSpJ"
# Pointの子クラスCircleの定義その2
class Circle2(Point):
    # Circleはインスタンス生成時に引数x,y,rを持つ
    def __init__(self, x, y, r):
        # xとyは、親クラスの属性として設定
        super().__init__(x, y)
        # rは、Circleの属性として設定
        self.r = r

    # draw関数は、子クラス独自に円の描画を行う
    def draw(self):
        # 円の描画
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)


# %% id="8-XPnRryPSpJ"
# クラスCircle2からインスタンス変数c2_1を生成する
c2_1 = Circle2(1, 0, 2)

# %% id="DKmZfGcMPSpJ" colab={"base_uri": "https://localhost:8080/", "height": 533} outputId="76feacea-7cc3-4e57-c6e7-bfe9902cc42d"
# p1, p2, c2_1 のそれぞれのfraw関数を呼び出す
ax = plt.subplot()
p1.draw()
p2.draw()
c2_1.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


# %% [markdown] id="4i0C8PMqPSpJ"
# 親のdarw関数の代わりに子のdraw関数が呼ばれたことがわかる  
# では、この関数と親の関数を両方呼びたいときはどうしたらいいか

# %% id="6IEbSPH8PSpJ"
# Pointの子クラスCircleの定義その3
class Circle3(Point):
    # Circleはインスタンス生成時に引数x,y,rを持つ
    def __init__(self, x, y, r):
        # xとyは、親クラスの属性として設定
        super().__init__(x, y)
        # rは、Circleの属性として設定
        self.r = r

    # Circleのdraw関数は、親の関数呼び出しの後で、円の描画も独自に行う
    def draw(self):
        # 親クラスのdraw関数呼び出し
        super().draw()

        # 円の描画
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)


# %% id="rmFwn3pCPSpJ"
# クラスCircle3からインスタンス変数c3_1を生成する
c3_1 = Circle3(1, 0, 2)

# %% id="AzhTYVYYPSpJ" colab={"base_uri": "https://localhost:8080/", "height": 533} outputId="60f5ccd4-97ea-4a49-fac1-ae8c3f8ac181"
# p1, p2, c3_1 のそれぞれのfraw関数を呼び出す
ax = plt.subplot()
p1.draw()
p2.draw()
c3_1.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


# %% [markdown] id="f9iMASK1PSpJ"
# 無事、両方を呼び出すことができた

# %% [markdown] id="eTmbHTnQPSpK"
# ## 1.6 インスタンスを関数として呼び出し可能にする

# %% id="rx8DIFfzPSpK"
# 関数クラスHの定義
class H:
    def __call__(self, x):
        return 2*x**2 + 2


# %% id="PmLLY2RwPSpK" colab={"base_uri": "https://localhost:8080/"} outputId="b6e83437-d301-4446-b43d-96250ae53d2f"
# hが関数として動作することを確認する

# numpy配列としてxの定義
x = np.arange(-2, 2.1, 0.25)
print(x)

# Hクラスのインスタンスとしてhを生成
h = H()

# 関数hの呼び出し
y = h(x)
print(y)

# %% id="aLJPaaFmPSpK" colab={"base_uri": "https://localhost:8080/", "height": 526} outputId="88649f46-86db-4f6d-e19b-f32eed0c9ced"
# グラフ描画
plt.plot(x, y)
plt.show()

# %% id="eRZh0Nv-PSpK"
