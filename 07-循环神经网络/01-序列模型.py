#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/18 15:41
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
from matplotlib import pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

plt.figure(figsize=(6, 3))
plt.plot(time, x)
plt.xlabel('time')
plt.ylabel('x_t')
plt.xlim([1, 1000])
plt.show()

# 样本：feature ={ x(t-4),x(t-3),x(t-2),x(t-1) }    label = x(t)
tau = 4
features = torch.zeros((T - tau, tau))  # torch.Size([996, 4])

for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape(-1, 1)

# print(x[:10])
# #tensor([ 0.1807, -0.1638, -0.1580,  0.1344, -0.1227,  0.0158,  0.3137, -0.1409, 0.0030, -0.0519])
# print(features[:5])
# print(labels[:5])
'''
tensor([[ 0.1807, -0.1638, -0.1580,  0.1344],
        [-0.1638, -0.1580,  0.1344, -0.1227],
        [-0.1580,  0.1344, -0.1227,  0.0158],
        [ 0.1344, -0.1227,  0.0158,  0.3137],
        [-0.1227,  0.0158,  0.3137, -0.1409]])
tensor([[-0.1227],
        [ 0.0158],
        [ 0.3137],
        [-0.1409],
        [ 0.0030]])
'''

batch_size = 16
num_train = 600
lr = 0.01
epochs = 10


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


train_loader = load_array((features[:600], labels[:600]), batch_size, is_train=True)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on", device)

#训练模型
def train(net,train_loader,loss,optimizer,epochs):
    for epoch in range(epochs):
        net.to(device)
        total_loss = 0
        for x, y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            l = loss(logits,y)
            l.backward()
            optimizer.step()

            total_loss += l
        print("epoch:%d,loss:%f" % (epoch+1,total_loss/len(train_loader)))

train(net,train_loader,loss,optimizer,epochs)


# 模型预测并显示结果
preds = net(features)
plt.figure(figsize=(6,3))
plt.plot(time,x)
plt.plot(time[tau:], preds.detach().numpy())
plt.xlim([1,1000])
plt.xlabel('time')
plt.ylabel('x')
plt.legend(['data', '1-step preds'])
plt.show()

# d2l.plot([time,time[tau:]],[x.detach().numpy(),preds.detach().numpy()],
#          'time','x',legend=['data','preds'],xlim=[1,1000],figsize=(6,3))
# d2l.plt.show()
#
# multistep_preds = torch.zeros(T)
# multistep_preds[:num_train+tau] = x[:num_train+tau]
#
# for i in range(num_train+tau,T):
#     multistep_preds[i] = net(multistep_preds[i-tau:i].reshape((1,-1)))
#
# d2l.plot([time,time[tau:],time[num_train+tau:]],
#          [x.detach().numpy(),preds.detach().numpy(),multistep_preds[num_train+tau:].detach().numpy()],
#          'time','x',legend=['data','1-step-preds','multistep-preds'],xlim=[1,1000],figsize=(6,3))
# d2l.plt.show()
#
# max_steps = 64
#
#
# features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# # 列 `i` (`i` < `tau`) 是来自 `x` 的观测
# # 其时间步从 `i + 1` 到 `i + T - tau - max_steps + 1`
# for i in range(tau):
#     features[:, i] = x[i:i + T - tau - max_steps + 1]
#
# # 列 `i` (`i` >= `tau`) 是 (`i - tau + 1`)步的预测
# # 其时间步从 `i + 1` 到 `i + T - tau - max_steps + 1`
# for i in range(tau, tau + max_steps):
#     features[:, i] = net(features[:, i - tau:i]).reshape(-1)
#
# steps = (1, 4, 16, 64)
# d2l.plot([time[tau + i - 1:T - max_steps + i] for i in steps],
#          [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time',
#          'x', legend=[f'{i}-step preds'
#                       for i in steps], xlim=[5, 1000], figsize=(6, 3))
# d2l.plt.show()
