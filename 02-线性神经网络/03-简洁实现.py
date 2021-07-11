#!/usr/bin/python3
"""
    Time    : 2021/4/26 20:13
    Author  : 春晓
    Software: PyCharm
"""

import numpy as np
import torch
import random
from torch.utils import data
from d2l import torch as d2l

#1.生成数据集
def gene_data(w, b, num_example):
    ''' Y = Xw+b+噪声，w={2，-3.4}，b=4.2'''
    x = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
# d2l.synthetic_data()和上一节的 gene_data()一样
features, labels = gene_data(true_w, true_b, 1000)

#2.小批量读取数据集
def load_array(data_arrays,batch_size,shuffle=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)  # 将tensor数据包装成dataset
    return data.DataLoader(dataset,batch_size,shuffle=shuffle)  # 将dataset包装成管道

batch_size = 10
data_loader = load_array((features,labels),batch_size)

#3.定义模型
from torch import nn
net = nn.Sequential(nn.Linear(2,1))

#4.初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)


#5.定义损失函数
loss = nn.MSELoss()

#6.定义优化算法
optimizer = torch.optim.SGD(net.parameters(),lr=0.03)

#7.训练
epochs = 10
for epoch in range(epochs):
    for x,y in data_loader:
        logits = net(x)
        l = loss(logits,y)  # torch.Size([]) 使用 pytorch中的损失函数，loss会自动对小批量求均值
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    # 记录每一个epoch训练完的loss值
    l = loss(net(features),labels)
    print("epoch:%d,loss:%f" % (epoch+1,l))
'''
epoch:1,loss:0.000206
epoch:2,loss:0.000107
epoch:3,loss:0.000107
epoch:4,loss:0.000106
epoch:5,loss:0.000106
epoch:6,loss:0.000107
epoch:7,loss:0.000106
epoch:8,loss:0.000106
epoch:9,loss:0.000106
epoch:10,loss:0.000106
'''


w = net[0].weight.data
b = net[0].bias.data
print("w:", w, '\n', "true_w", true_w)
print("b:", b, '\n', "true_b", true_b)
print("w的估计误差:{}\nb的估计误差:{}".format(true_w - w.reshape(true_w.shape), true_b - b))
'''
w: tensor([[ 1.9994, -3.3995]]) 
true_w tensor([ 2.0000, -3.4000])
b: tensor([4.2004]) 
true_b 4.2
w的估计误差:tensor([ 0.0006, -0.0005])
b的估计误差:tensor([-0.0004])
'''
