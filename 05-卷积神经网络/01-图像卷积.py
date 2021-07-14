#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/8 9:56
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn


# 1.互相关运算
def corr2d(x,k):
    '''计算二维互相关运算'''
    h,w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i:i+h,j:j+w]*k).sum()
    return y

x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
y = corr2d(x,k)
print(y)
'''
tensor([[19., 25.],
        [37., 43.]])
'''


#2.卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x,self.weight)+self.bias

