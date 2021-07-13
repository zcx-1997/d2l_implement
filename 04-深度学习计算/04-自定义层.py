#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/6 21:24
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torch.nn import functional as F

# 1.构造一个没有任何参数的层，实现减去均值的操作
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self,x):
        return x - x.mean()

layer = CenteredLayer()
out = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print(out)
#tensor([-2., -1.,  0.,  1.,  2.])

net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
out = net(torch.rand(4,8))
print(out.mean())
# tensor(4.6566e-09, grad_fn=<MeanBackward0>)


# 2.带参数的自定义层
class MyLinear(nn.Module):
    def __init__(self,in_units, out_units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units,out_units))
        self.bias = nn.Parameter(torch.randn(out_units,))

    def forward(self,x):
        linear = torch.matmul(x,self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5,3)
out = dense(torch.rand(2,5))
print(out)
'''
tensor([[0.3379, 0.0000, 2.7569],
        [0.0000, 0.0000, 1.6288]])
'''

net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
out = net(torch.rand(2,64))
print(out)
'''
tensor([[0.],
        [0.]])
'''

