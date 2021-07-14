#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/9 21:26
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torchsummary import summary
from toolFunctions.timer import Timer
from matplotlib import pyplot as plt

from toolFunctions.fashionMnist import load_data_fashion_mnist, sum_right, accuracy


# 1.Lenet模型
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


net1 = nn.Sequential(Reshape(),
                     nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                     nn.AvgPool2d(kernel_size=2, stride=2),
                     nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                     nn.AvgPool2d(kernel_size=2, stride=2),
                     nn.Flatten(),
                     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                     nn.Linear(120, 84), nn.Sigmoid(),
                     nn.Linear(84, 10)
                     )

# 观察一下每一层的输出
x = torch.rand((1, 1, 28, 28), dtype=torch.float32)
for layer in net1:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape:', x.shape)
'''
Reshape output shape:     torch.Size([1, 1, 28, 28])
Conv2d output shape:      torch.Size([1, 6, 28, 28])
Sigmoid output shape:     torch.Size([1, 6, 28, 28])
AvgPool2d output shape:   torch.Size([1, 6, 14, 14])
Conv2d output shape:      torch.Size([1, 16, 10, 10])
Sigmoid output shape:     torch.Size([1, 16, 10, 10])
AvgPool2d output shape:   torch.Size([1, 16, 5, 5])
Flatten output shape:     torch.Size([1, 400])
Linear output shape:      torch.Size([1, 120])
Sigmoid output shape:     torch.Size([1, 120])
Linear output shape:      torch.Size([1, 84])
Sigmoid output shape:     torch.Size([1, 84])
Linear output shape:      torch.Size([1, 10])
'''

# 2.Lenet实战训练
print("===================================================")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on ", device)

batch_size = 256
lr = 0.7
epochs = 10
train_loader, test_loader = load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                    nn.Linear(120, 84), nn.Sigmoid(),
                    nn.Linear(84, 10)
                    )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

# print(net)
summary(net, (1, 28, 28))

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)

net.to(device)

timer = Timer()
for epoch in range(epochs):

    timer.start()
    train_loss,train_acc, test_acc = 0, 0, 0
    #模型训练
    net.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = net(x)
        l = loss(logits, y)
        l.backward()
        optimizer.step()

        train_loss += l
        train_acc += accuracy(logits,y)

    #模型评估
    num_test = 0
    net.eval()
    for x,y in test_loader:
        x, y = x.to(device), y.to(device)
        test_acc += sum_right(net(x),y)
        num_test += len(y)

    if (epoch+1) % 2 == 0:
        print("epoch{}, train_loss={:.3f}, train_acc={:.3f}, test_acc={:.3f},time={:.5f}".format(
            epoch+1,train_loss/len(train_loader),train_acc/len(train_loader),test_acc/num_test,timer.stop()))
        #time=25.37158




