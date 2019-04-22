### coding: UTF-8
# MountainCarにおいて、Q学習を進めていき、
# その結果をgif出力する。
# Windowsで実行したのでimagemagickが必要。
# gif出力の参考：https://book.mynavi.jp/manatee/detail/id=88961

import numpy as np
import matplotlib.pyplot as plt

f=open('reward.csv')
line=f.readlines()
f.close()

nums=[i for i in range(10001)]
rewards =[float(i) for i in line ]
# for i in range(10000):
#     rewards.append(int(line[i]))

plt.plot(nums,rewards)
plt.show()