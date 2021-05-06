#!/usr/bin/python3
"""
    Time    : 2021/4/26 20:13
    Author  : 春晓
    Software: PyCharm
"""

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# d2l.synthetic_data()和上一节的 gene_data()一样
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 小批量读取数据集
def load_array(data_arrays,batch_size,shuffle=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)  # 将tensor数据包装成dataset
    return data.DataLoader(dataset,batch_size,shuffle=shuffle)  # 将dataset包装成管道

batch_size = 10
data_loader = load_array((features,labels),batch_size)

# 定义模型
from torch import nn
net = nn.Sequential(nn.Linear(2,1))

# 初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)


# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(),lr=0.03)

# 训练
epochs = 3
for epoch in range(epochs):
    for x,y in data_loader:
        l = loss(net(x),y)  # torch.Size([]) 使用pytorch中的损失函数，loss会自动对小批量求均值
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    # 记录每一个epoch训练完的loss值
    l = loss(net(features),labels)
    print("epoch:%d,loss:%f" % (epoch+1,l))


w = net[0].weight.data
b = net[0].bias.data
print("w:",w)
print("b:",b)
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)

