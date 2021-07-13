# -*- coding: utf-8 -*-

"""
@Time : 2021/7/13
@Author : Lenovo
@File : 03-权重衰减
@Description : 
"""
import torch
import numpy as np
from torch.utils import data
from torch import nn

from matplotlib import pyplot as plt


n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05  # torch.Size([200, 1])

features = torch.randn((n_train + n_test, num_inputs))  # torch.Size([120, 200])
labels = torch.matmul(features, true_w) + true_b  # torch.Size([120, 1])
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

train_features, test_features = features[:n_train, :], features[n_train:,
                                                       :]  # torch.Size([20, 200])  torch.Size([100, 200])
train_labels, test_labels = labels[:n_train], labels[n_train:]  # torch.Size([20, 1])  torch.Size([100, 1])

batch_size = 4
train_db = data.TensorDataset(train_features, train_labels)
train_loader = data.DataLoader(train_db, batch_size=batch_size, shuffle=True)


def train_decay(wd=0):
    epoches = 100
    lr = 0.003

    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=wd)

    train_ls, test_ls = [], []
    for epoch in range(epoches):

        net.train()
        train_loss = 0
        test_loss = 0
        for x,y in train_loader:
            logits = net(x)
            optimizer.zero_grad()
            l = loss(logits,y)
            l.backward()
            optimizer.step()

        train_loss = loss(net(train_features),train_labels)
        train_ls.append(train_loss)

        net.eval()
        test_loss = loss(net(test_features),test_labels)
        test_ls.append(test_loss)

        if (epoch+1) % 5 == 0:
            print("epoch{}: train_loss={:.5f}, test_loss={:.5f}".format(epoch+1,train_loss,test_loss))

    plt.plot(list(range(epoches)),train_ls)
    plt.plot(list(range(epoches)),test_ls)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train','test'])
    plt.show()

if __name__ == '__main__':
    train_decay()
    print("=============================")
    train_decay(wd=3)