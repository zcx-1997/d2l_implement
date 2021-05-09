#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/9 21:26
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

from d2l import torch as d2l

class Reshape(nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

net = nn.Sequential(Reshape(),
                    nn.Conv2d(1,6, kernel_size=5, padding=2),nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2,stride=2),
                    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2,stride=2),
                    nn.Flatten(),
                    nn.Linear(16*5*5,120),nn.Sigmoid(),
                    nn.Linear(120,84),nn.Sigmoid(),
                    nn.Linear(84,10)
                    )

x = torch.rand((1,1,28,28),dtype=torch.float32)
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:',x.shape)


batch_size = 256
lr = 0.1
epochs = 10
train_loader,test_loader = d2l.load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr)

d2l.train_ch3(net,train_loader,test_loader,loss,epochs,optimizer)
