# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：d2l_implement -> txt_process
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/18 下午5:23
@Description        ：
==================================================
"""

import re
import math
import torch
import random
import collections
from torch import nn
from torch.nn import functional as F


def read_time_machine():
    '''load timemachine.txt'''
    with open('../data/timemachine.txt','r',encoding='UTF-8') as f:
        lines = f.readlines()  # 返回一个列表，每行是一个字符串
        # 对列表中的每个字符串，将非英文字母字符用空格代替，将大写字母转换成小写并去掉两端的空白，返回仍是一个列表
        re_lines = [re.sub('[^A-Za-z]+',' ', line).strip().lower() for line in lines]
    return re_lines


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符标记。"""
    if token == 'word':
        return [line.split() for line in lines]  # 一个二维列表，每个字符串变成一个单词列表
    elif token == 'char':
        return [list(line) for line in lines]   # 一个二维列表，每个字符串变成一个字符列表
    else:
        print('错误：未知令牌类型：' + token)


def count_corpus(tokens):
    '''统计标记的频率'''
    if len(tokens) == 0 or isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]  # 将标记列表展平成一个列表
    return collections.Counter(tokens)  # 一个字典，标记：个数


class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens == None:
            tokens = []

        if reserved_tokens == None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        # 按频值降序排列
        self.token_freqs = sorted(counter.items(),key=lambda x:x[1],reverse=True)  # 降序
        self.unk = 0
        uniq_tokens = ['unk']+reserved_tokens  # 一个一维列表
        uniq_tokens += [token for token,freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token,self.token_to_idx = [],dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)  # 列表：索引-标记
            self.token_to_idx[token] = len(self.idx_to_token) - 1  # 字典：标记：索引

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):  # 返回tokens的索引，单个（值）或多个（列表）
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)

        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self,indices):  # 返回indices的token值，单个（值）或多个（列表）
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的标记索引列表和词汇表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每一个文本行，不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):  # 0-34,2,5
    """使用随机抽样生成一个小批量子序列。"""
    # 随机偏移量
    corpus = corpus[random.randint(0, num_steps - 1):]  # 0-34(/1-34/2-34/3-34/4-34)
    # 减去1，因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps  # 6
    # 长度为`num_steps`的子序列的起始索引列表
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  # [0, 5, 10, 15, 20, 25]
    random.shuffle(initial_indices)  # shuffle后的 initial_indices

    def data(pos):
        # 返回从`pos`开始的长度为`num_steps`的序列
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size  # 3
    for i in range(0, batch_size * num_batches, batch_size):  # for i in range(0,6,2)
        # 这里，`initial_indices`包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]  # [0,5],[10,15],[20,25]
        X = [data(j) for j in initial_indices_per_batch]  #[[0,1,2,3,4],[5,6,7,8,9]]
        Y = [data(j + 1) for j in initial_indices_per_batch]  #[[1,2,3,4,5],[6,7,8,9,10]]
        yield torch.tensor(X), torch.tensor(Y)


# 顺序采样
def seq_data_iter_sequential(corpus, batch_size, num_steps):  # 0-34,2,5
    """使用顺序分区生成一个小批量子序列。"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps-1)  # 0,1,2,3,4
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  #min=30
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1) #[2,15]
    num_batches = Xs.shape[1] // num_steps  # 3
    for i in range(0, num_steps * num_batches, num_steps):  # [0,5,10]
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器。"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)  # 需修改
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词汇表。"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,
                              max_tokens)
    return data_iter, data_iter.vocab


# 预测：传入一个字符串，生成之后的内容
def predict(prefix, num_preds, net, vocab, device):
    """在`prefix`后⾯⽣成新字符。 """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # [3]: "t"
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
        #[3,5,13,2,1,3,10,4,22,2,12,12,2,10,1]: "time traveller "
    for _ in range(num_preds):  # 预测`num_preds`步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])



def grad_clippling(net,theta):
    if isinstance(net,nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


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
            l.backward()
            grad_clippling(net, 1)
            optimizer.step()
        else:
            l.backward()
            optimizer(batch_size=1)
        total_loss += l * y.numel()
        num_tokes += y.numel()
    return math.exp(total_loss / num_tokes)


def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():  # 不构建计算图，操作不会被track
        for param in params:
            param -= lr * param.grad / batch_size  # 计算的损失是一个批量样本的总和,进行一下平均
            param.grad.zero_()

def train_epochs(net, train_data, vocab, lr, epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(),lr)
    else:
        optimizer = lambda batch_size: sgd(net.params,lr,batch_size)
    predict1 = lambda prefix: predict(prefix, 50, net, vocab, device)
    for epoch in range(epochs):
        ppl = train_epoch(net, train_data, loss, optimizer, device, use_random_iter)

        if (epoch + 1) % 1 == 0:
            print(predict1("time traveller "))
    print("困惑度：{}".format(ppl))
    print(predict1("time traveller "))
    print(predict1("traveller "))


class My_RNN(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(My_RNN, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        x = F.one_hot(inputs.T.long(), self.vocab_size)
        x = x.to(torch.float32)
        y, state = self.rnn(x, state)

        out = self.linear(y.reshape((-1, y.shape[-1])))
        return out, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))
