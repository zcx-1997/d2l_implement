#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/17 14:54
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1conv=False,strides=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)  # 覆盖，节约内存

    def forward(self,x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.relu(y)

blk = Residual(3, 3)
x = torch.rand(4, 3, 6, 6)
y = blk(x)
print(y.shape)
# torch.Size([4, 3, 6, 6])

b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                   nn.BatchNorm2d(64),nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                   )

def res_block(in_channels,num_channels,num_resduals,first_block=False):
    blk = []
    for i in range(num_resduals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels,num_channels,use_1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2 = nn.Sequential(*res_block(64,64,2,first_block=True))
b3 = nn.Sequential(*res_block(64,128,2))
b4 = nn.Sequential(*res_block(128,256,2))
b5 = nn.Sequential(*res_block(256,512,2))

net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),nn.Linear(512,10))

summary(net,(1,96,96))
