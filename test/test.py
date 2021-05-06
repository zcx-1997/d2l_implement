
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/6 8:54
    Author  : 春晓
    Software: PyCharm
"""

import torch
import numpy as np

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

before = id(x)
x[:] = x+y  # 或 a += b
print(x)
print(id(x) == before)
# True

x = torch.tensor([[1,2],[2,3]])
print(x.shape)

x = torch.arange(6).reshape(2,3)
a = torch.arange(3)
print('x',x)
print('a',a)
print(np.dot(x,a))
print(torch.mv(x,a))