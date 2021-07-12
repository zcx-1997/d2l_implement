#!/usr/bin/python3
"""
    Time    : 2021/4/27 9:47
    Author  : 春晓
    Software: PyCharm
"""

import torch
from torch.utils import data
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


# 1.加载数据集
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


# 2.初始化模型参数
num_imputs = 28 * 28
num_outputs = 10
w = torch.normal(0, 0.01, size=(num_imputs, num_outputs), requires_grad=True)  # 784,10
b = torch.zeros(num_outputs, requires_grad=True)  # 10


# 3.定义softmax操作
def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition


# 测试 softmax操作
# x = torch.normal(0, 1, (2, 5))
# x_prob = softmax(x)
# print(x_prob)
# '''
# tensor([[0.2520, 0.2035, 0.1899, 0.3331, 0.0215],
#         [0.5210, 0.0369, 0.0567, 0.1213, 0.2640]])
# '''
# print(x_prob.sum(1))
# '''
# tensor([1.0000, 1.0000])
# '''


# 定义模型

# 4.定义模型
def net(x):
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)


# 5.定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # log(y_hat[[0,1,...,255], y])


# 测试交叉熵损失函数
# y = torch.tensor([0, 2])  # 两个样本的标签
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1],y])  #tensor([0.1000, 0.5000])
# # y_hat[[0,1],[0,2]] == [y_hat[0,0],y_hat[1,2]]
# print(cross_entropy(y_hat, y))  # tensor([2.3026, 0.6931])

# 6.定义优化器
def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():  # 不构建计算图，操作不会被track
        for param in params:
            param -= lr * param.grad / batch_size  # 计算的损失是一个批量样本的总和,进行一下平均
            param.grad.zero_()


def optimizer(lr, batch_size):
    return sgd([w, b], lr, batch_size)


# 7.定义计算准确率函数
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

# print(sum_right(y_hat,y))  #1.0
# print(accuracy(y_hat,y))  #0.5


# 8.模型训练
def train(net, train_loader, loss, optimizer, lr, epochs):
    """训练模型 epochs个周期。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for x, y in train_loader:
            logits = net(x)
            l = loss(logits, y)
            if isinstance(optimizer, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                total_loss += l
                total_acc += accuracy(logits, y)
            else:
                # 使用自定义的优化器和损失函数
                l.sum().backward()  # 累加再反向传播
                optimizer(lr, x.shape[0])
                total_loss += l.sum() / len(y)
                total_acc += accuracy(logits, y)
        print("epoch{:d}: loss={:.5f}, acc:{:.5f}.".format(
            epoch + 1, total_loss / len(train_loader), total_acc / len(train_loader)))


#9.模型测试
def test(net, test_loader, loss):
    # 将模型设置为评估模式
    if isinstance(net, torch.nn.Module):
        net.eval()

    total_loss = 0
    total_acc = 0
    total_num = 0
    for x, y in test_loader:
        logits = net(x)
        l = loss(logits, y)
        total_loss += l.sum()
        total_acc += sum_right(logits, y)
        total_num += len(y)
    print("test: loss={:.5f}, acc={:.5f}".format(total_loss / total_num, total_acc / total_num))


#10.预测并显示图片
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

def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 显示预测的结果（前5个图片）
def predict(net,test_loader,num=5):
    for x,y in test_loader:
        break
    labels = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [(label + '\n' + pred) for label,pred in zip(labels,preds)]
    show_images(x[0:num].reshape((num,28,28)),1,num,titles=titles[0:num],scale=2)

if __name__ == '__main__':
    batch_size = 256
    lr = 0.1
    epochs = 3

    train_loader, test_loader = load_data_fashion_mnist(batch_size)
    loss = cross_entropy
    train(net, train_loader, loss, optimizer, lr, epochs)
    test(net, test_loader, loss)
    predict(net, test_loader)
