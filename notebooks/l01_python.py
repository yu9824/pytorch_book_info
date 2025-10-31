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

# %% [markdown] id="8PrD4L_92-EM"
# ## 講座1 Python  入門

# %% [markdown] id="PyYQrUX82-EP"
# #### 基本型

# %% id="ck6vpPgO2-EQ"
# 整数型
# 数値表現が整数の場合、代入先変数は自動的に整数型になります。
a = 1

# 浮動小数点数型
# 数値表現に小数点が含まれると、代入先変数は自動的に浮動小数点型になります。
b = 2.0

# 文字列型
# 文字列はシングルクオート(')で囲みます。
# あるいはダブルクオート(")でもいいです。
c = 'abc'

# ブーリアン型
# True または False と取る変数の型です。
d = True

# %% [markdown] id="47rgVwXz2-EY"
# #### print関数とtype関数

# %% id="wPuoMzM42-EY"
# 整数型変数aの値とtype
print(a)
print(type(a))

# %% id="cIu3RkVY2-Ef"
# 浮動小数点数型変数bの値とtype
print(b)
print(type(b))

# %% id="cDyOLlVE2-Ep"
# 文字列型変数cの値とtype'
print(c)
print(type(c))

# %% id="-eCkBPUd2-Ez"
# ブーリアン型変数dの値とtype'
print(d)
print(type(d))

# %% [markdown] id="HMcMwYOc2-E7"
# #### リスト

# %% id="yrRfmqkD2-E8"
# リストの定義
l = [1, 2, 3, 5, 8, 13]

# リストの値とtype
print(l)
print(type(l))

# %% [markdown] id="Hn9AiARl2-FH"
# #### リストの要素数

# %% id="BRJXtFFQ2-FI"
# リストの要素数
print(len(l))

# %% [markdown] id="HwysVIUG2-FV"
# #### リストの要素参照

# %% id="DK2_14rD2-FV"
# リストの要素参照

# 最初の要素
print(l[0])

# 3番目の要素
print(l[2])

# 最後の要素 (こういう指定方法も可能)
print(l[-1])

# %% [markdown] id="gQLlKCZgEMLI"
# #### 部分リスト参照1

# %% id="8RYCaj51EMLJ"
# 部分リスト インデックス:2以上 インデックス: 5未満
print(l[2:5])

# 部分リスト インデックス:0以上 インデックス: 3未満
print(l[0:3])

# 開始インデックスが0の場合は省略可
print(l[:3])

# %% [markdown] id="vYfz451iEMLJ"
# #### 部分リスト参照2

# %% id="gteronENEMLJ"
# 部分リスト インデックス:4以上最後まで
# リストの長さを求める
n = len(l) 
print(l[4:n])

# 最終インデックスが最終要素の場合は省略可
print(l[4:])

# 後ろから2つ
print(l[-2:])

#最初も最後も省略するとリスト全体になる
print(l[:])

# %% [markdown] id="0BcEDGw52-Fg"
# #### タプル

# %% id="clsh-h9U2-Fj"
# タプルの定義
t = (1, 2, 3, 5, 8, 13)

# タプルの値表示
print(t)

# タプルの型表示
print(type(t))

# タプルの要素数
print(len(t))

# タプルの要素参照
print(t[1])

# %% id="Ab5jH3-B2-Fn"
t[1] = 1

# %% id="0H7QX-AS2-Fq"
x = 1
y = 2
z = (x, y)
print(type(z))

# %% id="U3hYst__2-Ft"
a, b = z
print(a)
print(b)

# %% [markdown] id="vWpZdRag2-GJ"
# ### 辞書

# %% [markdown] id="qKh9XKnF2-GK"
# #### 辞書の定義

# %% id="2jPrrpfP2-GK"
# 辞書の定義
my_dict = {'yes': 1, 'no': 0}

# print文の結果
print(my_dict)

# type関数の結果
print(type(my_dict))

# %% [markdown] id="toUV79K72-GR"
# #### 辞書の参照

# %% id="20DLuIRv2-GS"
# キーから値を参照

# key= 'yes'で検索
value1 = my_dict['yes']
print(value1)

# key='no'で検索
value2 = my_dict['no']
print(value2)

# %% [markdown] id="ePA6aK2I2-GV"
# #### 辞書への項目追加

# %% id="nCsydOXH2-GW"
# 辞書への項目追加
my_dict['neutral'] = 2

# 結果確認
print(my_dict)

# %% [markdown] id="7A6JEafr2-Ga"
# ### 制御構造

# %% [markdown] id="83WIdtTd2-Ga"
# #### ループ処理

# %% id="5SqhNLCZ2-Gb"
# ループ処理

# リストの定義
list4 = ['One', 'Two', 'Three', 'Four']

# ループ処理
for item in list4:
    print(item)

# %% id="t-lTQuz22-Gh"
# range関数を使ったループ処理

for item in range(4):
    print(item)

# %% id="QkzvKe852-Gl"
# 引数2つのrange関数

for item in range(1, 5):
    print(item)

# %% id="OLqB-ql82-Gq"
# 辞書とループ処理

# items関数
print(my_dict.items())

# items関数を使ったループ処理

for key, value in my_dict.items():
    print(key, ':', value )

# %% [markdown] id="Lh4fff352-Gt"
# #### if文

# %% id="Zol5399N2-Gt"
# if文のサンプル
for i in range(1, 5):
    if i % 2 == 0:
        print(i, 'は偶数です')
    else:
        print(i, 'は奇数です')


# %% [markdown] id="fIXxsX3Y2-G5"
# ####  関数

# %% id="WKW7xi332-G5"
# 関数の定義例1
def square(x):
    p2 = x * x
    return p2

# 関数の呼び出し例1
x1 = 13
r1 = square(x1)
print(x1, r1)


# %% id="EnWVLp8o2-G_"
#  関数の定義例2
def squares(x):
    p2 = x * x
    p3= x * x * x
    return (p2, p3)

# 関数の呼び出し例2
x1 = 13
p2, p3 = squares(x1)
print(x1, p2, p3)

# %% [markdown] id="-lBmLd-D2-HC"
# #### ライプラリの導入

# %% id="coRwycFH2-HD"
# 日本語化ライブラリ導入
# !pip install japanize-matplotlib | tail -n 1

# %% [markdown] id="pIlNIJ5b2-HF"
# #### import文

# %% id="4256qlzn2-HG"
# 必要ライブラリのimport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# matplotlib日本語化対応
import japanize_matplotlib

# データフレーム表示用関数
from IPython.display import display

# %% [markdown] id="HwEohvnZ2-HT"
# #### ワーニング非表示

# %% id="dBVuCfby2-HT"
# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

# %% [markdown] id="sBjLfxG02-HY"
# #### 数値の整形表示

# %% id="kRbJQtIu2-HZ"
# f文字列の表示
a1 = 1.0/7.0
a2 = 123

str1 = f'a1 = {a1}   a2 = {a2}'
print(str1)

# %% id="U-J22Vwg2-Hc"
# f文字列の詳細オプション

# .4f 小数点以下4桁の固定小数点表示
# 04 整数を0詰め4桁表示
str2 = f'a1 = {a1:.4f}  a2 = {a2:04}'
print(str2)

# 04e 小数点以下4桁の浮動小数点表示
# #x 整数の16進数表示
str3 = f'a1 = {a1:.04e}  a2 = {a2:#x}'
print(str3)

# %% id="6fN1g2E42-Hj"
