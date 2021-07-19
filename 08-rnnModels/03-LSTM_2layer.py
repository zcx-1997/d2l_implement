# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：d2l_implement -> 03-LSTM_2layer
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/19 下午7:52
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
vocab_size = len(vocab)
num_inputs, num_hiddens, num_layers = vocab_size, 256, 2

lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers)

net = My_RNN(lstm_layer,vocab_size)
met = net.to(device)
train_epochs(net,train_loader,vocab,lr,epochs,device)