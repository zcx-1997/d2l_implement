#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/5 19:30
    Author  : 春晓
    Software: PyCharm
"""
from torch.nn import functional as F
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(20,256),
    nn.ReLU(),
    nn.Linear(256,10)
)

x = torch.rand(2,20)
y = net(x)
print(y)

# 自定义类
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self, x):
        out = self.hidden(x)
        out = F.relu(self.out(out))
        return out

net = MLP()
y2 = net(x)
print(y2)