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

batch_size,num_steps= 32, 35

train_loader,vocab = d2l.load_data_time_machine(batch_size,num_steps)

x = torch.arange(0,10).reshape(2,5)  # batch_size,num_steps
print(F.one_hot(x.T,28).shape)  # num_steps,batch_size,len(vocab)

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

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

