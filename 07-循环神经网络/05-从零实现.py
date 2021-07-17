#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/21 16:16
    Author  : 春晓
    Software: PyCharm
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 1.独热编码
x = torch.arange(10).reshape(10, 1)  # torch.Size([10, 1])
# print(F.one_hot(x.T, 10).shape)  # torch.Size([1, 10, 10])
# print(F.one_hot(x.T, 10))
'''
tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]])
'''


# x = torch.arange(10).reshape(2, 5)  # batch_size, num_steps:(2,5)
# print(F.one_hot(x.T, 28).shape)  # (时间步长，批量大小，词汇表大小): torch.Size([5, 2, 28])

# 2.单隐藏层 RNN
# 训练语⾔模型时，输⼊和输出来⾃相同的词表
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 初始化隐藏状态，返回⼀个张量，全⽤ 0填充，形状为(批量⼤小, 隐藏单元数)
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # inputs: （时间步长，批量大小，词表大小）
    # outputs：(时间步数×批量⼤小, 词汇表⼤小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for x in inputs:  # x：（批量大小，词表大小）
        h = torch.tanh(torch.matmul(x, W_xh) + torch.matmul(H, W_hh) + b_h)
        y = torch.matmul(h, W_hq) + b_q
        outputs.append(y)
    outputs = torch.cat(outputs, dim=0)
    return outputs, (h,)


class RNN:
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, x, state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
batch_size, num_steps = 32, 35
num_hiddens = 512

train_loader, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# train_loader:(2,32,35),len(vocab)=28

# 3.检测一下网络输出
# 输出形状是(时间步数×批量⼤小, 词汇表⼤小)
# 隐藏状态形状保持不变，即(批量⼤小, 隐藏单元数)
x = torch.arange(10).reshape(2, 5)  # batch_size, num_steps:(2,5)

net = RNN(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
state = net.begin_state(x.shape[0], device)  # 初始化隐藏状态，0
# print(state[0].shape)  #torch.Size([2, 512]),
y, new_state = net(x.to(device), state)

print(y.shape)  # torch.Size([10, 28])
print(new_state[0].shape)  # torch.Size([2, 512])

# 查看下输出
x, y = next(iter(train_loader))  # torch.Size([32, 35]), torch.Size([32, 35])
y = y.T.reshape(-1)  # torch.Size([1120])


# 4.预测：传入一个字符串，生成之后的内容
def predict(prefix, num_preds, net, vocab, device):
    """在`prefix`后⾯⽣成新字符。 """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测`num_preds`步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 5.模型训练
def train_epoch(net, train_data, loss, optimizer, device, use_random_iter):
    '''训练一个epoch'''
    state = None
    total_loss, num_tokes = 0, 0
    for x, y in train_data:
        # 处理state
        if state is None or use_random_iter:
            # 在第⼀次迭代或使⽤随机抽样时初始化`state`
            state = net.begin_state(batch_size=x.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()  # 从计算图中分离出来
            else:
                for s in state:
                    s.detach_()

        y = y.T.reshape(-1)  ##torch.Size([batch_size * num_steps])
        x, y = x.to(device), y.to(device)
        y_hat, state = net(x, state)
        l = loss(y_hat, y.long()).mean()  # tensor.long() 转换成 long型
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            l.backword()
            optimizer.step()
        else:
            l.backward()
            optimizer(batch_size=1)
        total_loss += l * y.numel()
        num_tokes += y.numel()
    return math.exp(total_loss / num_tokes)


def train_epochs(net, train_data, vocab, lr, epochs, device, use_random_iter=False):

    loss = nn.CrossEntropyLoss()

    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(),lr)

    else:
        optimizer = lambda batch_size: d2l.sgd(net.params,lr,batch_size)
    predict1 = lambda prefix: predict(prefix, 50, net, vocab, device)
    for epoch in range(epochs):
        ppl = train_epoch(net, train_data, loss, optimizer, device, use_random_iter)

        if (epoch + 1) % 1 == 0:
            print(predict1("time traveller"))

    print("困惑度：{}".format(ppl))
    print(predict1("time traveller"))
    print(predict1("traveller"))


epochs = 500
lr = 1
train_epochs(net, train_loader, vocab, lr, epochs, device)
