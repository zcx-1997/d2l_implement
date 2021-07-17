#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/10 16:53
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torchsummary import summary
from toolFunctions.train_gen import train
from toolFunctions.fashionMnist import load_data_fashion_mnist

net = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Flatten(),
    nn.Linear(256*5*5,4096),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,10)
)



# 观察一下每一层的输出
# print(net)
# summary(net,(1,224,224))

# net1 = nn.Sequential(
#     nn.Conv2d(1,32,kernel_size=3,padding=1),nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2,stride=2),
#     nn.Conv2d(32,64,kernel_size=3,padding=1),nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2,stride=2),
#     nn.Conv2d(64,128,kernel_size=3,padding=1),nn.ReLU(),
#     nn.Conv2d(128,256,kernel_size=3,padding=1),nn.ReLU(),
#     nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2,stride=2),
#     nn.Flatten(),
#     nn.Linear(256*3*3,4096),nn.ReLU(),nn.Dropout(0.5),
#     nn.Linear(4096,2048),nn.ReLU(),nn.Dropout(0.5),
#     nn.Linear(2048,10)
# )
# summary(net1,(1,28,28))


batch_size = 256
lr = 0.1
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on", device)
train_loader,test_loader = load_data_fashion_mnist(batch_size,resize=224)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr)
train(net,train_loader,test_loader,loss,optimizer,epochs,device)
#time=45.89147