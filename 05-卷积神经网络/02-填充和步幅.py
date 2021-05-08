#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/8 10:47
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

def pad_conv2d(conv2d,x):
    # 这里的（1，1）表示批量大小和通道数都是1
    x = x.reshape((1,1)+x.shape)
    y = conv2d(x)
    # 省略前两个维度：批量大小和通道
    return y.reshape(y.shape[2:])

conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
x = torch.rand(8,8)
y = pad_conv2d(conv2d,x)
print(y.shape)

# torch.Size([8, 8])


conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
y = pad_conv2d(conv2d, x)
print(y.shape)

# torch.Size([8, 8])

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
print(pad_conv2d(conv2d, x).shape)
# torch.Size([4, 4])

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,5), padding=(0,1), stride=(3,4))
print(pad_conv2d(conv2d, x).shape)
# torch.Size([2, 2])