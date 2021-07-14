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
from toolFunctions.train_gen import train
from toolFunctions.fashionMnist import load_data_fashion_mnist


# vgg block:图片高宽减半,通道数in_channel-->out_channel
def vgg_block(num_convs, in_channels, out_channels):
    '''卷积保持图片高宽不变，通道数in_channel-->out_channel，池化图片高宽减半,通道数不变'''
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    #卷积部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         #全连接部分
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 10))

#VGG-11：8+3
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
vgg_11 = vgg(conv_arch)
summary(vgg_11, (1, 224, 224))

batch_size = 256
lr = 0.1
epochs = 10

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
train_loader,test_loader = load_data_fashion_mnist(batch_size,resize=224)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg_11.parameters(),lr)
train(vgg_11,train_loader,test_loader,loss,optimizer,epochs,device)


