#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/16 17:06
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn
from torchsummary import summary
from d2l import torch as d2l


def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels

    net = nn.Sequential(*conv_blks,nn.Flatten(),
                        nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
                        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
                        nn.Linear(4096,10)
                        )

    return  net

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
net = vgg(conv_arch)
summary(net,(1,224,224))

