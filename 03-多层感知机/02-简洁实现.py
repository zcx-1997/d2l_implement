#!/usr/bin/python3
"""
    Time    : 2021/4/28 20:04
    Author  : 春晓
    Software: PyCharm
"""
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

from d2l import torch as d2l

batch_size = 256
def load_data_fashion_mnist(batch_size,resize=None):
    ''' resize调整图片大小'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='../data',train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data',train=False,
                                                   transform=trans,
                                                   download=True)
    train_d = data.DataLoader(mnist_train,batch_size,shuffle=True)
    test_d = data.DataLoader(mnist_test,batch_size,shuffle=True)
    return train_d,test_d

train_loader,test_loader = load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.Linear(256,10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 保证 y_hat与 y的数据类型一致
    # print(cmp.dtype)  # torch.bool
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 模型训练
def train_epoch_ch3(net, train_loader, loss, optimizer):  #@save
    """训练模型一个迭代周期（定义见第3章）。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    for x, y in train_loader:
        y_hat = net(x)
        l = loss(y_hat, y)
        # print(x.shape)  # torch.Size([256, 1, 28, 28])
        # print(y.shape)  # torch.Size([256])
        # print(y_hat.shape)  # torch.Size([256, 10])
        # print(l.shape)  # torch.Size([])
        if isinstance(optimizer, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel())
            # float(l) * len(y) pytorch会自动对loss取均值
            # print(y.size())  #torch.Size([256])
            # print(y.numel()) # 256
            # print(y.size().numel()) # 256
        else:
            # 使用自定义的优化器和损失函数
            l.sum().backward()  # 累加再反向传播
            optimizer(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_loader, test_loader, loss, num_epochs, optimizer):  #@save
    """训练模型（定义见第3章）。"""
    # # d2l.Animator是一个在动画中绘制数据的实用程序类
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_loader, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_loader)
        print("epoch:%d,loss:%f,acc:%f\ntest acc:%f" % (epoch+1,train_metrics[0],train_metrics[1],test_acc))
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc

num_epochs = 10
train_ch3(net, train_loader, test_loader, loss, num_epochs, optimizer)
