#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/6 19:30
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
x = torch.rand(2, 4)
y = net(x)


#1.参数访问
print(net[2].state_dict())
#OrderedDict([('weight', tensor([[ 0.3303,  0.1171,  0.3146, -0.0960, 0.2041, -0.2837, -0.3512,  0.1375]])), ('bias', tensor([-0.1550]))])

print(type(net[2].bias))  #<class 'torch.nn.parameter.Parameter'>
print(net[2].bias)
'''
Parameter containing:
tensor([-0.1550], requires_grad=True)
'''
print(net[2].bias.data)  #tensor([-0.1550])
print(net.state_dict()['2.bias'].data)  #tensor([-0.1550])
print(net[2].bias.grad)  #None

# 访问第一个全连接层的参数和
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# 访问所有层
print(*[(name, param.shape) for name, param in net.named_parameters()])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(2):
        net.add_module('block %d' % i, block1())
    return net


# 嵌套块模型
Rnet = nn.Sequential(block2(), nn.Linear(4, 1))
y3 = Rnet(x)
print(Rnet)
'''
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)

'''

print(Rnet[0][1][0].bias.data)
#tensor([ 0.1112, -0.4316,  0.3184, -0.2286, -0.3111, -0.4240,  0.3753, -0.1090])


#2.参数初始化

#内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias[0])
#tensor([ 6.5577e-03,  5.0457e-03, -8.0769e-05, -4.7349e-03]) tensor(0., grad_fn=<SelectBackward>)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias[0])
#tensor([1., 1., 1., 1.]) tensor(0., grad_fn=<SelectBackward>)

#第一层xavier，第二层常初始化为常数2
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_2(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 2)


net[0].apply(xavier)
net[2].apply(init_2)
print(net[0].weight.data[0])
print(net[2].weight.data)
#tensor([ 0.3335, -0.6552,  0.5218, -0.4231])
#tensor([[2., 2., 2., 2., 2., 2., 2., 2.]])


# 自定义参数初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init ", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
# Init  weight torch.Size([8, 4])
# Init  weight torch.Size([1, 8])

net.apply(my_init)
print(net[0].weight[:2])
# tensor([[ 0.0000,  6.1537, -8.1242,  0.0000],
#         [-6.1325, -8.9075, -9.2868, -9.2114]], grad_fn=<SliceBackward>)

# 在多个层间共享参数
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
net(x)

# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
#tensor([True, True, True, True, True, True, True, True])