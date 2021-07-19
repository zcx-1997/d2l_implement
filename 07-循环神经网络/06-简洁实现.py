# -*- coding: utf-8 -*-

"""
@Time : 2021/7/15
@Author : Lenovo
@File : 06-简洁实现
@Description : 
"""

import torch
from torch import nn
from torch.nn import functional as F

from toolFunctions.txt_process import load_data_time_machine, predict, \
    train_epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, num_steps = 32, 35
train_loader, vocab = load_data_time_machine(batch_size, num_steps)
# state = torch.zeros()

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)


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


net = My_RNN(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
print(predict('time traveller ', 10, net, vocab, device))

print("training on:", device)
epochs, lr = 500, 1
train_epochs(net, train_loader, vocab, lr, epochs, device)
