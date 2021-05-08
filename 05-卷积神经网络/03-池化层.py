#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/8 16:36
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

x = torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
pool2d = nn.MaxPool2d(3)
print(pool2d(x).shape)
# torch.Size([1, 1, 1, 1])

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(x).shape)
# torch.Size([1, 1, 2, 2])

pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
print(pool2d(x).shape)
# torch.Size([1, 1, 3, 2])