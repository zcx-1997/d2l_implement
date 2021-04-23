#!/usr/bin/python3
"""
    Time    : 2021/4/16 15:09
    Author  : 春晓
    Software: PyCharm
"""
import torch

x = torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

print(x.shape)
# torch.Size([12])
print(x.size())
# torch.Size([12])