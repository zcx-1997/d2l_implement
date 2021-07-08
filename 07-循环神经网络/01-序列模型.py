#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/18 15:41
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from d2l import torch as d2l

T = 1000
time = torch.arange(1,T+1,dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0,0.2,(T,))
d2l.plot(time,[x],'time','x',xlim=[1,1000],figsize=(6,3))
d2l.plt.show()

# 样本：feature=（x0,x1,x2,x3） label=(x4)
tau = 4
features = torch.zeros((T-tau,tau))
print(features.shape)
for i in range(tau):
    features[:,i] = x[i:T-tau+i]
labels = x[tau:].reshape(-1,1)

# print(x[:10])
# print(features[:5])
# print(labels[:5])
# print(x[4:9])

batch_size = 16
num_train = 600
lr = 0.01
epochs = 10

train_loader = d2l.load_array((features[:600],labels[:600]),batch_size,is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    net.apply(init_weights)
    return net

net = get_net()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr)

def train(net,train_loader,loss,optimizer,epochs):
    for epoch in range(epochs):
        total_loss = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            logits = net(x)
            l = loss(logits,y)
            l.backward()
            optimizer.step()

            total_loss += l
        print("epoch:%d,loss:%f" % (epoch+1,total_loss/len(train_loader)))

train(net,train_loader,loss,optimizer,epochs)

preds = net(features)
d2l.plot([time,time[tau:]],[x.detach().numpy(),preds.detach().numpy()],
         'time','x',legend=['data','preds'],xlim=[1,1000],figsize=(6,3))
d2l.plt.show()

multistep_preds = torch.zeros(T)
multistep_preds[:num_train+tau] = x[:num_train+tau]

for i in range(num_train+tau,T):
    multistep_preds[i] = net(multistep_preds[i-tau:i].reshape((1,-1)))

d2l.plot([time,time[tau:],time[num_train+tau:]],
         [x.detach().numpy(),preds.detach().numpy(),multistep_preds[num_train+tau:].detach().numpy()],
         'time','x',legend=['data','1-step-preds','multistep-preds'],xlim=[1,1000],figsize=(6,3))
d2l.plt.show()

max_steps = 64


features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列 `i` (`i` < `tau`) 是来自 `x` 的观测
# 其时间步从 `i + 1` 到 `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i:i + T - tau - max_steps + 1]

# 列 `i` (`i` >= `tau`) 是 (`i - tau + 1`)步的预测
# 其时间步从 `i + 1` 到 `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1:T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time',
         'x', legend=[f'{i}-step preds'
                      for i in steps], xlim=[5, 1000], figsize=(6, 3))
d2l.plt.show()