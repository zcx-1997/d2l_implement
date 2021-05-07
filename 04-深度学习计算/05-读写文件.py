#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/6 21:54
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torch.nn import functional as F

# 保存和读取单个张量
x = torch.arange(4)
torch.save(x,'x-file')

x1 = torch.load('x-file')
print(x1)

# 保存和读取张量列表
y = torch.zeros(4)
torch.save([x,y],'x-y-file')
x2,y2 = torch.load('x-y-file')
print(x2,y2)

# 保存和读取从字符串映射到张量的字典
mydict = {'x':x, 'y':y}
torch.save(mydict,'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))

net = MLP()
x = torch.rand(2,20)
y = net(x)

torch.save(net.state_dict(),'mlp.params')

net2 = MLP()
net2.load_state_dict(torch.load('mlp.params'))
y2 = net2(x)
print(y == y2)
