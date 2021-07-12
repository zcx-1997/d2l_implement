#!/usr/bin/python3
"""
    Time    : 2021/4/27 14:54
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

from toolFunctions.timer import Timer
timer = Timer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
def load_data_fashion_mnist(batch_size, resize=None):
    ''' 加载fashion—mnist数据集，resize调整图片大小'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = datasets.FashionMNIST(root='../data', train=True,
                                        transform=trans,
                                        download=True)
    mnist_test = datasets.FashionMNIST(root='../data', train=False,
                                       transform=trans,
                                       download=True)
    train_d = data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_d = data.DataLoader(mnist_test, batch_size, shuffle=True)
    return train_d, test_d


# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)



#计算准确率
def sum_right(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum())


def accuracy(y_hat, y):
    """计算准确率。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) / len(y)


# 模型训练
def train(net, train_loader, loss, optimizer, epochs):
    """训练模型 epochs个周期。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            l = loss(logits, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l
            total_acc += accuracy(logits, y)

        print("epoch{:d}: loss={:.5f}, acc:{:.5f}.".format(
            epoch + 1, total_loss / len(train_loader), total_acc / len(train_loader)))


#测试
def test(net, test_loader, loss):
    # 将模型设置为评估模式
    if isinstance(net, torch.nn.Module):
        net.eval()

    total_loss = 0
    total_acc = 0
    total_num = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = net(x)
        l = loss(logits, y)
        total_loss += l.sum()
        total_acc += sum_right(logits, y)
        total_num += len(y)
    print("test: loss={:.5f}, acc={:.5f}".format(total_loss / total_num, total_acc / total_num))


if __name__ == '__main__':
    batch_size = 256
    lr = 0.1
    epochs = 10

    train_loader, test_loader = load_data_fashion_mnist(batch_size)

    # 定义损失和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)
    net = net.to(device)
    timer.start()
    train(net, train_loader, loss, optimizer, epochs)
    test(net, test_loader, loss)
    print("time:",timer.stop())
    #time: 70.64251565933228
