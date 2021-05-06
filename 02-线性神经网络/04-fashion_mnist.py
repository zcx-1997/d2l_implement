#!/usr/bin/python3
"""
    Time    : 2021/4/27 8:47
    Author  : 春晓
    Software: PyCharm
"""

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt


# 读取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='../data',train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data',train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)

# print(len(mnist_train),len(mnist_test))     # 60000 10000
# print(len(mnist_train[0]))                  # 2
# print(mnist_train[0][0].shape)              # feature (1,28,28)
# print(mnist_train[0][1])                    # label 9
# print(mnist_train[0])


# 索引与标签的转化
def get_fashin_mnist_labels(labels):  # label:0-9
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt','sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 可视化样本
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):  #@save
    figsize = (num_cols*scale,num_rows*scale)  # 设置图片大小
    fig,axes = plt.subplots(num_rows,num_cols,figsize=figsize)  # 设置子图
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


x,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
print(x.shape,y.shape)
show_images(x.reshape(18,28,28),2,8,titles=get_fashin_mnist_labels(y))


# 读取小批量
batch_size = 256


# 单进程读取时间
timer = d2l.Timer()
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
for x,y in train_iter:
    continue
print("time(s):",timer.stop())

# # 多进程读取
# 4个进程
# def get_dataloader_worker():
#     return 4
#
# timer.start()
# train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_worker())
# for x,y in train_iter:
#     continue
# print("time(s):",timer.stop())

# 整合所有组件
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

# 28*28 --> 64*64
train_loader, test_loader = load_data_fashion_mnist(32, resize=64)
for x, y in train_loader:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break


