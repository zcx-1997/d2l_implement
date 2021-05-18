#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/17 15:33
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torchsummary import summary

def conv_block(in_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),nn.ReLU(),
        nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self,num_convs,in_channels,num_channels):
        super(DenseBlock, self).__init__()

        layer = []
        for i in range(num_convs):
            layer.append(conv_block((num_channels * i + in_channels), num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self,x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x,y),dim=1)
        return x

blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

def transition_block(in_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# `num_channels`为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(b1, *blks, nn.BatchNorm2d(num_channels), nn.ReLU(),
                    nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(),
                    nn.Linear(num_channels, 10))

summary(net,(1,96,96))