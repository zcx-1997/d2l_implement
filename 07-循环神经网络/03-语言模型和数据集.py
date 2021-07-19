# -*- coding: utf-8 -*-
"""
    Time    : 2021/5/19 15:04
    Author  : 春晓
    Software: PyCharm
"""
import random
import torch
from matplotlib import pyplot as plt

from toolFunctions.txt_process import read_time_machine, tokenize,load_corpus_time_machine
from toolFunctions.txt_process import Vocab



lines = read_time_machine()
tokens = tokenize(lines)

# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行连接到一起
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
print(vocab.token_freqs[:10])
'''
[('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), 
 ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)]
'''

freqs = [freq for token,freq in vocab.token_freqs]
# plt.plot(freqs)
# plt.xlabel('token')
# plt.ylabel('frequency:n(x)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


# # 二元语法
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
# print(bigram_vocab.token_freqs[:10])
'''
[(('of', 'the'), 309), (('in', 'the'), 169), (('i', 'had'), 130), 
 (('i', 'was'), 112), (('and', 'the'), 109), (('the', 'time'), 102),
 (('it', 'was'), 99), (('to', 'the'), 85), (('as', 'i'), 78), 
 (('of', 'a'), 73)]
'''


# 三元语法
trigram_tokens = [
    triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
# print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

plt.plot(freqs)
plt.plot(bigram_freqs)
plt.plot(trigram_freqs)
plt.xlabel('token')
plt.ylabel('frequency:n(x)')
plt.xscale('log')
plt.yscale('log')
plt.legend(['one','two','three'])
plt.show()


print("==============================================================")
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

my_seq = list(range(35))
for epoch in range(2):
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)
'''
X:  tensor([[15, 16, 17, 18, 19],
        [ 0,  1,  2,  3,  4]]) 
Y: tensor([[16, 17, 18, 19, 20],
        [ 1,  2,  3,  4,  5]])
        ...
'''

print("===========================================================")
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

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

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