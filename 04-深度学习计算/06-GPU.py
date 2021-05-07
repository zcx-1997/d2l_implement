#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/7 11:36
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

x = torch.tensor([1, 2, 3])
print(x.device)
x.to(device)
print(x)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device)
