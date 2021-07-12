#!/usr/bin/python3
"""
    Time    : 2021/4/27 8:47
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch.utils import data
from torchvision import datasets, transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt

# 1.读取数据集
mnist_train = datasets.FashionMNIST(root='../data', train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
mnist_test = datasets.FashionMNIST(root='../data', train=False,
                                   transform=transforms.ToTensor(),
                                   download=True)

print(len(mnist_train), len(mnist_test))  # 60000 10000
print(mnist_train[0][0].shape)  # torch.Size([1, 28, 28]):features[0]
print(mnist_train[0][1])  # 9:label[0]


# 2.实现通过数字索引来获取文本标签
def get_fashin_mnist_labels(labels):  # label:0-9
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 3.创建一个函数，实现可视化样本
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


x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
print(x.shape, y.shape)  # torch.Size([18, 1, 28, 28]) torch.Size([18])
show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashin_mnist_labels(y))


# 4.整合所有组件
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


# 28*28 --> 64*64
train_loader, test_loader = load_data_fashion_mnist(32, resize=64)
x, y = next(iter(train_loader))
print(x.shape, x.dtype, y.shape, y.dtype)
# torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
