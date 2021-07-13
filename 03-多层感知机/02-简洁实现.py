#!/usr/bin/python3
"""
    Time    : 2021/4/28 20:04
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch import nn
from toolFunctions.timer import Timer
from matplotlib import pyplot as plt

from toolFunctions.fashionMnist import load_data_fashion_mnist, sum_right, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.Linear(256,10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)


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

        test_acc = 0
        test_num = 0

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            l = loss(logits, y)

            test_acc += sum_right(logits, y)
            test_num += len(y)
        print("test: acc={:.5f}".format(test_acc / test_num))
        test_acc_list.append(test_acc/test_num)

    plt.plot(list(range(epochs)), loss_list)
    plt.plot(list(range(epochs)), acc_list)
    plt.plot(list(range(epochs)), test_acc_list)
    plt.xlabel('epoch')
    plt.legend(['train loss', 'train acc','test acc'])
    plt.show()




if __name__ == '__main__':

    timer = Timer()

    batch_size = 256
    lr = 0.1
    epochs = 10

    train_loader, test_loader = load_data_fashion_mnist(batch_size)

    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)

    net = net.to(device)
    timer.start()
    train(net, train_loader, test_loader, loss, optimizer, epochs)
    # test(net, test_loader, loss)

    print("time:", timer.stop())
    # time: 115.34698343276978
