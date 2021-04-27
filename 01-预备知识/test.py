#!/usr/bin/python3
"""
    Time    : 2021/4/27 19:34
    Author  : 春晓
    Software: PyCharm
"""
import torch

a = torch.tensor([[[0.0,1,2],[1,1,1]]])
b = torch.ones(((2,3)),dtype=torch.float)
# b[:,1] = 0
print(a)
print(a.shape)
print(b)
print(b.shape)
c = torch.matmul(a,b)
print(c)
print(c.shape)
# print(torch.matmul(b,a))

# (2,3)      （3，）---》（2，）
#（2,3）     （3,1）---》（2,1）
#（1,2,3）   （3，）---》（1,2）
#（1,2,3）   （3,1）---》（1,2,1）
# (1,2,3)    (2,3)---> error
