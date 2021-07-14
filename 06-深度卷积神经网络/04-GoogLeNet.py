#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/17 10:38
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn
from torchsummary import summary
from toolFunctions.train_gen import train
from toolFunctions.fashionMnist import load_data_fashion_mnist


class Inception(nn.Module):
    '''c1,c2,c3,c4 是每条路径的输出通道数'''
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception, self).__init__(**kwargs)

        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)

        self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        #在通道维度上连接
        return torch.cat((p1,p2,p3,p4),dim=1)


b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                   nn.ReLU(),nn.MaxPool2d(kernel_size=3,padding=1))
b2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=1),nn.ReLU(),
                   nn.Conv2d(64,192,kernel_size=3,padding=1),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b3 = nn.Sequential(Inception(192,64,(96,128),(16,32),32),
                   Inception(256,128,(128,192),(32,96),64),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b4 = nn.Sequential(Inception(480,192,(96,208),(16,48),64),
                   Inception(512,160,(112,224),(24,64),64),
                   Inception(512,128,(128,256),(24,64),64),
                   Inception(512,112,(144,288),(32,64),64),
                   Inception(528,256,(160,320),(32,128),128),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                   )

b5 = nn.Sequential(Inception(832,256,(160,320),(32,128),128),
                   Inception(832,384,(192,384),(48,128),128),
                   nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()
                   )
net = nn.Sequential(b1,b2,b3,b4,b5,nn.Linear(1024,10))

summary(net,(1,96,96))

batch_size = 256
lr = 0.1
epochs = 10

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
print("training on", device)
train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=96)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)
train(net, train_loader, test_loader, loss, optimizer, epochs, device)