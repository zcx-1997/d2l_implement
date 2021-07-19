# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：d2l_implement -> 02-LSTM
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/19 下午7:49
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
lstm_layer = nn.LSTM(num_inputs,num_hiddens)

net = My_RNN(lstm_layer,vocab_size)
met = net.to(device)
train_epochs(net,train_loader,vocab,lr,epochs,device)