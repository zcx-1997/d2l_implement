# -*- coding: utf-8 -*-
"""
    Time    : 2021/5/18 15:41
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt

# 1.data
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

plt.figure(figsize=(6, 3))
plt.plot(time, x)
plt.xlabel('time')
plt.ylabel('x_t')
plt.xlim([1, 1000])
plt.show()

# 2.sample ：feature ={ x(t-4),x(t-3),x(t-2),x(t-1) } ; label = x(t)
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

#3.train
batch_size = 16
num_train = 600
lr = 0.01
epochs = 10


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


train_loader = load_array((features[:num_train], labels[:num_train]),
                          batch_size, is_train=True)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


net = get_net()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on", device)


# 训练模型
def train(net, train_loader, loss, optimizer, epochs, device):
    for epoch in range(epochs):
        net.to(device)
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            l = loss(logits, y)
            l.backward()
            optimizer.step()

            total_loss += l
        print("epoch:%d,loss:%f" % (epoch + 1, total_loss / len(train_loader)))


train(net, train_loader, loss, optimizer, epochs, device)

#4.模型预测并显示结果(1-step)
features = features.to(device)
preds = net(features)
# plt.figure(figsize=(6, 3))
# plt.plot(time, x)
# plt.plot(time[tau:], preds.detach().cpu().numpy())
# plt.xlim([1, 1000])
# plt.xlabel('time')
# plt.ylabel('x')
# plt.legend(['data', '1-step preds'])
# plt.show()

#5.predict(mul-step)
multistep_preds = torch.zeros(T).to(device)

multistep_preds[:num_train+tau] = x[:num_train+tau].to(device)
for i in range(num_train+tau,T):
    multistep_preds[i] = net(multistep_preds[i-tau:i].reshape((1,-1)).to(device))

plt.figure(figsize=(6, 3))
plt.plot(time, x)
plt.plot(time[tau:], preds.detach().cpu().numpy())
plt.plot(time[num_train+tau:], multistep_preds[num_train+tau:].detach().cpu().numpy())
plt.xlim([1, 1000])
plt.xlabel('time')
plt.ylabel('x')
plt.legend(['data', '1-step preds','mul-step'])
plt.show()

