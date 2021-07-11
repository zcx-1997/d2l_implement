#!/usr/bin/python3
"""
    Time    : 2021/4/26 18:38
    Author  : 春晓
    Software: PyCharm
"""

import random
import torch
from matplotlib import pyplot as plt


# 1.生成数据集
def gene_data(w, b, num_example):  # @save
    ''' Y = Xw+b+噪声，w={2，-3.4}，b=4.2'''
    x = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = gene_data(true_w, true_b, 1000)
print(features.shape)  # torch.Size([1000, 2])
print(labels.shape)  # torch.Size([1000, 1])
# 查看第一个样本
print(features[0])  # tensor([-0.0784, -1.2610])
print(labels[0])  # tensor([8.3166])

'''
# 显示生成的数据集
plt.figure(figsize=(7,5))
ax = plt.axes(projection='3d')  #设置三维轴
ax.scatter3D(features[:,0].detach().numpy(),features[:,1].detach().numpy(),
             labels.detach().numpy())
plt.xlabel('feature_0')
plt.ylabel('feature_1')
ax.set_zlabel('labels')
plt.show()
'''


# 2.小批量读取数据集
def data_loader(features, labels, batch_size):
    '''yield使data_loader成为了一个生成器，可以视为一个管道'''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# batch_size = 2
# for x,y in data_loader(features,labels,batch_size):
#     #一个批次，minibatch=2
#     print(x,'\n',y)
#     '''
#     tensor([[-0.0079,  2.1846],
#         [-0.7978, -0.6048]])
#     tensor([[-3.2493],
#         [ 4.6413]])
#     '''
#     break

# 3.初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 4.定义模型
def linreg(x, w, b):
    '''线性回归模型'''
    return torch.matmul(x, w) + b


# 5.定义损失函数
def squared_loss(y_hat, y):
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 6.定义优化器
def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():  # 不构建计算图，操作不会被track
        for param in params:
            param -= lr * param.grad / batch_size  # 计算的损失是一个批量样本的总和,进行一下平均
            param.grad.zero_()


# 7.模型训练
lr = 0.03
epochs = 10
batch_size = 10

net = linreg
loss = squared_loss

for epoch in range(epochs):
    for x, y in data_loader(features, labels, batch_size):
        logits = net(x, w, b)  # torch.Size([10, 1])
        l = loss(logits, y)  # torch.Size([10, 1])
        l.sum().backward()  # 保证标量，将小批量的loss值累加后再backward
        sgd([w, b], lr, batch_size)  # 梯度更新

    # 记录每一个epoch训练完的loss值
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print("epoch: %d, loss:%f" % (epoch + 1, train_l.mean()))
'''
epoch: 1, loss:0.040198
epoch: 2, loss:0.000144
epoch: 3, loss:0.000050
epoch: 4, loss:0.000050
epoch: 5, loss:0.000050
epoch: 6, loss:0.000050
epoch: 7, loss:0.000050
epoch: 8, loss:0.000050
epoch: 9, loss:0.000050
epoch: 10, loss:0.000050
'''

print("w:", w, '\n', "true_w", true_w)
print("b:", b, '\n', "true_b", true_b)
print("w的估计误差:{}\nb的估计误差:{}".format(true_w - w.reshape(true_w.shape), true_b - b))
'''
w: tensor([[ 2.0005],[-3.3993]], requires_grad=True) 
true_w tensor([ 2.0000, -3.4000])
b: tensor([4.1996], requires_grad=True) 
true_b 4.2
w的估计误差:tensor([-0.0005, -0.0007], grad_fn=<SubBackward0>)
b的估计误差:tensor([0.0004], grad_fn=<RsubBackward1>)
'''