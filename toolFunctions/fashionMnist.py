# -*- coding: utf-8 -*-

"""
@Time : 2021/7/12
@Author : Lenovo
@File : fashionMnist
@Description : 
"""

import torch
from torch.utils import data
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# 实现通过数字索引来获取文本标签
def get_fashin_mnist_labels(labels):  # label:0-9
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#创建一个函数，实现可视化样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)  # 设置图片大小
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)  # 设置子图
    # _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()  # 将子图（轴）组织成一个列表，使其可以通过索引来设置
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 不显示x轴
        ax.axes.get_yaxis().set_visible(False)  # 不显示y轴
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

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