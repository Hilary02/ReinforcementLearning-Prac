### coding: UTF-8
# 出力されたrewardsないし数字列をグラフにして表示する
# 引数に入力するファイルを記述することで指定できる
# 引数がない場合、実行ディレクトリのrewards.csvを読み込む

import sys
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv

if len(args) <= 1:
    f = open('reward.csv')
else :
    f = open(args[1])

line=f.readlines()
f.close()

nums=[i for i in range(len(line))]
rewards =[float(i) for i in line ]
# for i in range(10000):
#     rewards.append(int(line[i]))

plt.plot(nums,rewards)
plt.show()