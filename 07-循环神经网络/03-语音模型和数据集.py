#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/19 15:04
    Author  : 春晓
    Software: PyCharm
"""
import random
import torch
import re
from d2l import torch as d2l

def read_time_machine():
    with open('../data/timemechine.txt','r',encoding='UTF-8') as f:
        lines = f.readlines()
        return [re.sub('[^A-Za-z]+',' ', line).strip().lower() for line in lines]

lines = read_time_machine()
tokens = d2l.tokenize(lines)

# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行连接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token,freq in vocab.token_freqs]
d2l.plot(freqs,xlabel='token:x',ylabel='frequency:n(x)',xscale='log',yscale='log')
# d2l.plt.show()

# 二元语法
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元语法
trigram_tokens = [
    triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
# d2l.plt.show()


# 随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):  # 0-34,2,5
    """使用随机抽样生成一个小批量子序列。"""
    # 随机偏移量
    corpus = corpus[random.randint(0, num_steps - 1):]  # 0-34,1-34,2-34,3-34
    # 减去1，因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps  # 6
    # 长度为`num_steps`的子序列的起始索引列表
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  # [0, 5, 10, 15, 20, 25]

    random.shuffle(initial_indices)  # shuffle后的 initial_indices

    def data(pos):
        # 返回从`pos`开始的长度为`num_steps`的序列
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size  # 3
    for i in range(0, batch_size * num_batches, batch_size):  # 0,2,4
        # 这里，`initial_indices`包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]  # [0,5],[10,15],[20,25]
        X = [data(j) for j in initial_indices_per_batch]  # [2,5]
        Y = [data(j + 1) for j in initial_indices_per_batch]  # [2,5]
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
# for epoch in range(2):
#     for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#         print('X: ', X, '\nY:', Y)

# 顺序采样
def seq_data_iter_sequential(corpus, batch_size, num_steps):  # 0-34,2,5
    """使用顺序分区生成一个小批量子序列。"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)  # 2
    # offset = 2  # 2
    # 变成batch_size的倍数
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  # 30
    Xs = torch.tensor(corpus[offset:offset + num_tokens])   # 2-31
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])  # 3-32
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)  # [2,15]
    num_batches = Xs.shape[1] // num_steps  # 3
    for i in range(0, num_steps * num_batches, num_steps):  # [0,5,10]
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

class SeqDataLoader:
    """加载序列数据的迭代器。"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)  # 需修改
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词汇表。"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,
                              max_tokens)
    return data_iter, data_iter.vocab