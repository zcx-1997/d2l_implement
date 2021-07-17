#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/19 10:12
    Author  : 春晓
    Software: PyCharm
"""
import collections
import re

# from d2l import torch as d2l
#
# d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
#                                 '090b5e7e70c295757f55df93cb0a180b9691891a')

# 读取数据集
# def read_time_machine(): #@save
#     """Load the time machine dataset into a list of text lines."""
#     with open(d2l.download('time_machine'), 'r') as f:
#         lines = f.readlines()
#     return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def read_time_machine():
    with open('../data/timemachine.txt','r',encoding='UTF-8') as f:
        lines = f.readlines()  # 返回一个列表，每行是一个字符串
        # 对列表中的每个字符串，将非英文字母字符用空格代替，将大写字母转换成小写并去掉两端的空白，返回仍是一个列表
        re_lines = [re.sub('[^A-Za-z]+',' ', line).strip().lower() for line in lines]
    return re_lines

lines = read_time_machine()
print('text lines: %d' % len(lines))  #3221
print(lines[0])  #the time machine by h g wells
print(lines[10])  #twinkled and his usually pale face was flushed and animated the

# 标记化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符标记。"""
    if token == 'word':
        return [line.split() for line in lines]  # 一个二维列表，每个字符串变成一个单词列表
    elif token == 'char':
        return [list(line) for line in lines]   # 一个二维列表，每个字符串变成一个字符列表
    else:
        print('错误：未知令牌类型：' + token)

tokens = tokenize(lines)  # 二维列表
for i in range(11):
    print(tokens[i])
'''
['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
[]
[]
[]
[]
['i']
[]
[]
['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
'''

# 建立词汇表，并进行数字索引
def count_corpus(tokens):
    '''统计标记的频率'''
    if len(tokens) == 0 or isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]  # 将标记列表展平成一个列表
    return collections.Counter(tokens)  # 一个字典，标记：个数

# counter = count_corpus(tokens)
# print(counter.most_common(10))
# print(counter.items())  # 以列表元组的形式：[（key,value）,(,),...] 表示字典

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

vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])

print('words:', tokens[0])
print('indices:', vocab[tokens[0]])
print('tokens:',vocab.to_tokens(vocab[tokens[0]]))
'''
words: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
indices: [1, 19, 50, 40, 2183, 2184, 400]
tokens: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
'''

#整合到一起
print("===================每个字符映射到一个编码=====================")
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

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))  #170580 28
print(corpus[:20])
print(vocab.to_tokens(corpus[:20]))
'''
===================每个字符映射到一个编码=====================
170580 28
[3, 9, 2, 1, 3, 5, 13, 2, 1, 13, 4, 15, 9, 5, 6, 2, 1, 21, 19, 1]
['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 'm', 'a', 'c', 'h', 'i', 'n', 'e', ' ', 'b', 'y', ' ']
'''