#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/6 19:30
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
x = torch.rand(2,4)
y = net(x)

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net.state_dict()['2.bias'].data)

print(net[2].bias.grad)

# 访问第一个全连接层的参数和
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
# 访问所有层
print(*[(name,param.shape) for name,param in net.named_parameters()])



def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(2):
        net.add_module('block %d' % i,block1())
    return net

# 嵌套块模型
Rnet = nn.Sequential(block2(),nn.Linear(4,1))
y3 = Rnet(x)
print(Rnet)

print(Rnet[0][1][0].bias.data)


# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0],net[0].bias[0])

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0],net[0].bias[0])

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_2(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,2)

net[0].apply(xavier)
net[2].apply(init_2)
print(net[0].weight.data[0])
print(net[2].weight.data)


# 自定义参数初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init ", *[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])


# 在多个层间共享参数
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
net(x)

# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
