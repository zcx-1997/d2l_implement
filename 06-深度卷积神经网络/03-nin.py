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


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
    return net


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),

    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

summary(net, (1, 224, 224))

batch_size = 256
lr = 0.1
epochs = 10

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
print("training on", device)
train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=224)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)
train(net, train_loader, test_loader, loss, optimizer, epochs, device)
