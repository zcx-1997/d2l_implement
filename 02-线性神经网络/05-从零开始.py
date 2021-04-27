#!/usr/bin/python3
"""
    Time    : 2021/4/27 9:47
    Author  : 春晓
    Software: PyCharm
"""

import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch.utils import data
from d2l import torch as d2l
from torchvision import transforms

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


# # 整合所有组件,读取数据
# batch_size = 256
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

# train_loader,test_loader = load_data_fashion_mnist(batch_size)

# 初始化模型参数
# num_imputs = 28*28
# num_outputs = 10
# w = torch.normal(0,0.01,size=(num_imputs,num_outputs),requires_grad=True) # 784,10
# b = torch.zeros(num_outputs,requires_grad=True)  # 10

#　定义softmax操作
def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1,keepdim=True)
    return x_exp/partition

# a = torch.normal(0,1,(2,5))
# a_prob = softmax(a)
# print(a_prob)

# 定义模型
def net(x):
    return softmax(torch.matmul(x.reshape((-1,w.shape[0])),w)+b)

# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 定义优化器
# lr = 0.1
def optimizer(batch_size):
    return d2l.sgd([w, b], lr, batch_size)
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1],y])  # y[[0,1],[0,2]] == [y[0,0],y[1,2]]
# print(cross_entropy(y_hat, y))

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 保证 y_hat与 y的数据类型一致
    # print(cmp.dtype)  # torch.bool
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
    """训练模型一个迭代周期（定义见第3章）。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for x, y in train_loader:
        # 计算梯度并更新参数
        y_hat = net(x)  # b
        l = loss(y_hat, y)  # b
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

# num_epochs = 10
# train_ch3(net, train_loader, test_loader, cross_entropy, num_epochs, optimizer)


# 可视化样本
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):  #@save
    figsize = (num_cols*scale,num_rows*scale)
    _,axes = plt.subplots(num_rows,num_cols,figsize=figsize)
    # _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes
# 模型预测
def predict_ch3(net,test_loader,num=5):
    for x,y in test_loader:
        break
    labels = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [(label + '\\' + pred) for label,pred in zip(labels,preds)]
    # print(titles[0:num])
    show_images(x[0:num].reshape((num,28,28)),1,num,titles=titles[0:num],scale=2)

# predict_ch3(net,test_loader)

if __name__ == '__main__':
    batch_size = 256
    lr = 0.1
    num_epochs = 10
    train_loader, test_loader = load_data_fashion_mnist(batch_size)
    num_imputs = 28 * 28
    num_outputs = 10
    w = torch.normal(0, 0.01, size=(num_imputs, num_outputs), requires_grad=True)  # 784,10
    b = torch.zeros(num_outputs, requires_grad=True)  # 10

    train_ch3(net, train_loader, test_loader, cross_entropy, num_epochs, optimizer)
    predict_ch3(net, test_loader)