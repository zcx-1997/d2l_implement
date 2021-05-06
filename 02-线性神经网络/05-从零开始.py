#!/usr/bin/python3
"""
    Time    : 2021/4/27 9:47
    Author  : 春晓
    Software: PyCharm
"""

import torch
from matplotlib import pyplot as plt
from d2l import torch as d2l

# 一个累加器
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#　定义softmax操作
def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1,keepdim=True)
    return x_exp/partition

# 定义模型
def net(x):
    return softmax(torch.matmul(x.reshape((-1,w.shape[0])),w)+b)

# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # log(y_hat[[0,1,...,255], y])

# y = torch.tensor([0, 2])  # 两个样本的标签
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1],y])
# y_hat[[0,1],[0,2]] == [y_hat[0,0],y_hat[1,2]] = [0.1000, 0.5000]
# print(cross_entropy(y_hat, y))  # tensor([2.3026, 0.6931])

# 定义优化器
def optimizer(batch_size):
    return d2l.sgd([w, b], lr, batch_size)


# 定义计算准确率函数
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 保证 y_hat与 y的数据类型一致,cmp是bool值
    return float(cmp.type(y.dtype).sum())
# acc = accuracy(y_hat,y) / len(y)
# print(acc)

# 在测试数据集上测试 net的准确率
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
# test_acc = evaluate_accuracy(net,test_loader)
# print(test_acc)

# 模型训练
def train_epoch_ch3(net, train_loader, loss, optimizer):  #@save
    """训练模型一个迭代周期。"""

    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数

    for x, y in train_loader:
        y_hat = net(x)  # (256,10)
        l = loss(y_hat, y)  # (256,)
        # x.shape (256,1,28,28)
        # y.shape (256,)
        # w.shape (28*28,10)
        # b.shape (10,)
        # y_hat （256,10）
        # l （256,10）

        if isinstance(optimizer, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel()) # y.size().numel() = 256
        else:
            # 使用自定义的优化器和损失函数
            l.sum().backward()  # 累加再反向传播
            optimizer(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

# 训练模型并显示在测试集上的准确率
def train_ch3(net, train_loader, test_loader, loss, num_epochs, optimizer):  #@save
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_loader, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_loader)
        print("epoch:%d,loss:%f,acc:%f\ntest acc:%f" % (epoch+1,train_metrics[0],train_metrics[1],test_acc))

# 显示预测的结果（前5个图片）
def predict_ch3(net,test_loader,num=5):
    for x,y in test_loader:
        break
    labels = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [(label + '\n' + pred) for label,pred in zip(labels,preds)]
    d2l.show_images(x[0:num].reshape((num,28,28)),1,num,titles=titles[0:num],scale=2)

if __name__ == '__main__':
    batch_size = 256
    lr = 0.1
    num_epochs = 1

    train_loader, test_loader = d2l.load_data_fashion_mnist(batch_size)

    # 初始化模型参数
    num_imputs = 28 * 28
    num_outputs = 10
    w = torch.normal(0, 0.01, size=(num_imputs, num_outputs), requires_grad=True)  # 784,10
    b = torch.zeros(num_outputs, requires_grad=True)  # 10

    train_ch3(net, train_loader, test_loader, cross_entropy, num_epochs, optimizer)
    predict_ch3(net, test_loader)
