#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/17 14:54
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn

# from torchsummary import summary

from toolFunctions.train_gen import train
from toolFunctions.fashionMnist import load_data_fashion_mnist

#1.残差块
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

'''
#输入和输出一致
blk = Residual(3, 3)
x = torch.rand(4, 3, 6, 6)
y = blk(x)
print(y.shape)
# torch.Size([4, 3, 6, 6])

#改变通道数
blk2 = Residual(3, 5, use_1conv=True)
x = torch.rand(4, 3, 6, 6)
y2 = blk2(x)
print(y2.shape)
# torch.Size([4, 5, 6, 6])

#改变通道数,并减半高宽
blk2 = Residual(3, 5, use_1conv=True, strides=2)
x = torch.rand(4, 3, 6, 6)
y2 = blk2(x)
print(y2.shape)
# torch.Size([4, 5, 3, 3])
'''


print("==================================")
#2.残差模型

b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                   nn.BatchNorm2d(64),nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                   )

def res_block(in_channels,num_channels,num_resduals,first_block=False):
    blk = []
    for i in range(num_resduals):
        if i == 0 and not first_block:
            #除了第一个模块中的第一个残差块，其他模块中的第一个残差块将宽高减半
            blk.append(Residual(in_channels,num_channels,use_1conv=True,strides=2))
        else:
            #第一个模块中的第一个残差块和所有残差块中的第二残差块
            blk.append(Residual(num_channels,num_channels))
    return blk

b2 = nn.Sequential(*res_block(64,64,2,first_block=True))
b3 = nn.Sequential(*res_block(64,128,2))
b4 = nn.Sequential(*res_block(128,256,2))
b5 = nn.Sequential(*res_block(256,512,2))

#ResNet-18
net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),nn.Linear(512,10))

# summary(net,(1,224,224))
'''
Sequential output shape: torch.Size([1, 64, 56, 56])        b1  1
Sequential output shape: torch.Size([1, 64, 56, 56])        b2  4
Sequential output shape: torch.Size([1, 128, 28, 28])       b3  4
Sequential output shape: torch.Size([1, 256, 14, 14])       b4  4
Sequential output shape: torch.Size([1, 512, 7, 7])         b5  4
AdaptiveAvgPool2d output shape: torch.Size([1, 512, 1, 1])
Flatten output shape: torch.Size([1, 512])
Linear output shape: torch.Size([1, 10])                    fc 1
'''


batch_size = 32
lr = 0.5
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on", device)
train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=96)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)

resnet_18 = net
train(resnet_18, train_loader, test_loader, loss, optimizer, epochs, device)