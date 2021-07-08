#!/usr/bin/python3
"""
    Time    : 2021/4/27 14:54
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)


loss = nn.CrossEntropyLoss()  # 没有进行独热编码
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

if __name__ == '__main__':
    batch_size = 4
    lr = 0.1
    num_epochs = 1

    train_loader, test_loader = d2l.load_data_fashion_mnist(batch_size)
    for x,y in train_loader:
        print(x.shape)  # (4,1,28,28)
        print(y.shape)  # (4,)
        logits = net(x)
        print(logits.shape) # (4,10)
        print(logits,y)
        l = loss(logits,y)
        print(l.shape)

        break


    d2l.train_ch3(net, train_loader, test_loader, loss, num_epochs, optimizer)
    # d2l.predict_ch3(net, test_loader)