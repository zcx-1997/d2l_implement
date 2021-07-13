#!/usr/bin/python3
"""
    Time    : 2021/4/28 17:01
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from toolFunctions.timer import Timer
from matplotlib import pyplot as plt

from toolFunctions.fashionMnist import load_data_fashion_mnist, sum_right, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# w = torch.normal(0,0.01,size=(num_imputs,num_outputs),requires_grad=True) # 784,10
# b = torch.zeros(num_outputs,requires_grad=True)  # 10
w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [w1, b1, w2, b2]


# 定义激活函数
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = x.reshape((-1, num_inputs))
        h = relu(x @ w1 + b1)
        return h @ w2 + b2


# 模型训练
def train(net, train_loader, test_loader, loss, optimizer, epochs):
    """训练模型 epochs个周期。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    loss_list = []
    acc_list = []
    test_acc_list = []
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

        loss_list.append(total_loss / len(train_loader))
        acc_list.append(total_acc / len(train_loader))

        # 模型评估
        net.eval()

        test_loss = 0
        test_acc = 0
        test_num = 0

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            l = loss(logits, y)
            test_loss += l.sum()
            test_acc += sum_right(logits, y)
            test_num += len(y)
        print("test: loss={:.5f}, acc={:.5f}".format(test_loss / test_num, test_acc / test_num))
        test_acc_list.append(test_acc/test_num)

    plt.plot(list(range(epochs)), loss_list)
    plt.plot(list(range(epochs)), acc_list)
    plt.plot(list(range(epochs)), test_acc_list)
    plt.xlabel('epoch')
    plt.legend(['train loss', 'train acc','test acc'])
    plt.show()


# 测试
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

    timer = Timer()

    batch_size = 256
    lr = 0.1
    epochs = 10

    train_loader, test_loader = load_data_fashion_mnist(batch_size)

    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr)

    net = Net().to(device)
    timer.start()
    train(net, train_loader, test_loader, loss, optimizer, epochs)
    # test(net, test_loader, loss)

    print("time:", timer.stop())
    # time: 205.78307008743286
