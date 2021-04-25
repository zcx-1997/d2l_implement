#!/usr/bin/python3
"""
    Time    : 2021/4/25 11:13
    Author  : 春晓
    Software: PyCharm
"""

import torch

x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于 `x = torch.arange(4.0, requires_grad=True)`
print(x.grad)  # 默认None

y = 2*torch.dot(x,x)
y.backward()
print(x)
print(y)
print(x.grad)

# 在默认情况下，PyTorch会累积梯度，我们需要在backwa前清除之前的值
x.grad.zero_()
print(x.grad)
y = x.sum()
y.backward()
print(x.grad)

y = (x*x).sum()
print(y)


# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()  # Returns a new Tensor, detached from the current graph.The result will never require gradient.
z = u * x   # 将u看作常数，则x.grad = u

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad)

# python 控制流

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(),requires_grad=True)  # 标量
d = f(a)
d.backward()
print(a.grad==(d/a))

print(dir(torch))
print(help(torch.ones))