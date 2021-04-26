#!/usr/bin/python3
"""
    Time    : 2021/4/26 15:30
    Author  : 春晓
    Software: PyCharm
"""

import math
import time
import numpy as np
import torch
from d2l import torch as d2l

class Timer:  #@save
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

n = 100000
a = torch.ones(n)
b = torch.ones(n)

c = torch.zeros(n)
timer = Timer()
# for循环计算
for i in range(n):
    c[i] = a[i]+b[i]
print("time(s):",timer.stop())
# time(s): 1.2830934524536133

# 矢量化计算
timer.start()
d = a + b
print("time(s):",timer.stop())
# time(s): 0.0009987354278564453