# -*- coding: utf-8 -*-

"""
@Time : 2021/7/14
@Author : Lenovo
@File : train_gen
@Description : 
"""

from toolFunctions.timer import Timer
from toolFunctions.fashionMnist import sum_right, accuracy


def train(net,train_loader,test_loader,loss,optimizer,epochs,device):
    timer = Timer()
    for epoch in range(epochs):

        timer.start()
        train_loss, train_acc, test_acc = 0, 0, 0
        # 模型训练
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            l = loss(logits, y)
            l.backward()
            optimizer.step()

            train_loss += l
            train_acc += accuracy(logits, y)

        # 模型评估
        num_test = 0
        net.eval()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_acc += sum_right(net(x), y)
            num_test += len(y)

        if (epoch + 1) % 2 == 0:
            print("epoch{}, train_loss={:.3f}, train_acc={:.3f}, test_acc={:.3f},time={:.5f}".format(
                epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader), test_acc / num_test,
                timer.stop()))