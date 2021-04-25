#!/usr/bin/python3
"""
    Time    : 2021/4/16 15:09
    Author  : 春晓
    Software: PyCharm
"""
import torch

x = torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

print(x.shape)
x = x.reshape(3,4)
print(x.shape)

print(type(x))
print(x.size())
print(x.numel())
print(x.dtype)

x = x.reshape(2,2,3)
print(x.shape)
x = x.reshape(-1, 4)
print(x.shape)

x1 = torch.zeros((2,3))
x2 = torch.ones((2,3))
x3 = torch.rand([2,3])
x4 = torch.randn([2,3])
x5 = torch.tensor([[0,1,2],[3,4,5]])
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x**y,x//y)

a = torch.exp(torch.tensor(1))
print(a)
b = torch.log(a)
print(b)

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a+b)

before = id(a)
a = a + b
print(id(a) == before)

before = id(a)
a[:] = a+b  # 或 a += b
print(id(a) == before)

y1 = x.numpy()
y2 = torch.tensor(y1)
print(type(y1),type(y2))