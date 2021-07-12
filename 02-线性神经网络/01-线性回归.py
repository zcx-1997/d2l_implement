#!/usr/bin/python3
"""
    Time    : 2021/4/26 15:30
    Author  : 春晓
    Software: PyCharm
"""

import torch
from toolFunctions import timer


n = 100000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)

timer = timer.Timer()
timer.start()
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