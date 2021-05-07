#!/usr/bin/python3
"""
    Time    : 2021/4/28 20:04
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_loader,test_loader = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.Linear(256,10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)


num_epochs = 10
d2l.train_ch3(net, train_loader, test_loader, loss, num_epochs, optimizer)
