#!/usr/bin/python3
"""
    Time    : 2021/4/28 17:01
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_loader,test_loader = d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens = 784,10,256
# w = torch.normal(0,0.01,size=(num_imputs,num_outputs),requires_grad=True) # 784,10
# b = torch.zeros(num_outputs,requires_grad=True)  # 10
w1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params = [w1,b1,w2,b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x,a)

def net(x):
    x = x.reshape((-1,num_inputs))
    h = relu(x@w1+b1)
    return h@w2+b2

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params,lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_loader, test_loader, loss, num_epochs, optimizer)

