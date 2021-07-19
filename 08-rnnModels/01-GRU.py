# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：d2l_implement -> 01-GRU
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/19 下午7:36
@Description        ：
==================================================
"""
import torch
from torch import nn

from  toolFunctions.txt_process import load_data_time_machine,My_RNN,train_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 500
lr = 1
batch_size,num_steps = 32, 35
train_loader, vocab = load_data_time_machine(batch_size,num_steps)
num_hiddens = 256
vocab_size = len(vocab)
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs,num_hiddens)

net = My_RNN(gru_layer,vocab_size)
met = net.to(device)
train_epochs(net,train_loader,vocab,lr,epochs,device)


