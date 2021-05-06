#!/usr/bin/python3
"""
    Time    : 2021/4/26 18:38
    Author  : 春晓
    Software: PyCharm
"""

import random
import torch
from d2l import torch as d2l

# 生成数据集
def gene_data(w,b,num_example): #@save
    ''' y = 2x1 - 3.4x2 + 4.2 + 噪声'''
    x = torch.normal(0,1,(num_example,len(w)))
    y = torch.matmul(x,w)+b
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = gene_data(true_w, true_b,1000)
print(features[0])
print(labels[0])


# 小批量读取数据集
def data_loader(features,labels,batch_size):
    '''yield使data_loader成为了一个生成器，可以视为一个管道'''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

#　初始化模型参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# 定义模型
def linreg(x,w,b):
    '''
    :param x: batch*2
    :param w: 2
    :param b: 1
    :return: batch*1
    '''
    return torch.matmul(x,w)+b

# 定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

# 定义优化器
def sgd(params,lr,batch_size):  #@save
    with torch.no_grad():  # 不构建计算图，操作不会被track
        for param in params:
            param -= lr*param.grad / batch_size  # 计算的损失是一个批量样本的总和,进行一下平均
            param.grad.zero_()

#　模型训练
lr = 0.03
epochs = 3
batch_size = 10

# 模型和损失函数都是自己定义的函数
net = linreg
loss = squared_loss

for epoch in range(epochs):
    for x,y in data_loader(features,labels,batch_size):
        '''
        x: batch * 2
        y: batch * 1
        logit: batch * 1
        l: batch * 1
        '''
        logit = net(x,w,b)
        l = loss(net(x,w,b),y)
        l.sum().backward()  # 保证标量，将小批量的loss值累加后再backward
        sgd([w,b],lr,batch_size)  # 梯度更新

    #记录每一个epoch训练完的loss值
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print("epoch: %d, loss:%f" % (epoch+1,train_l.mean()))

print("w:",w)
print("b:",b)
print("w的估计误差:{}\nb的估计误差:{}".format(true_w - w.reshape(true_w.shape),true_b - b))
