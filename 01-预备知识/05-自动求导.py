#!/usr/bin/python3
"""
    Time    : 2021/4/25 11:13
    Author  : 春晓
    Software: PyCharm
"""

import torch

#1.一个简单的例子
x = torch.arange(4.0)
x.requires_grad_(True)
# 以上两句等价于 `x = torch.arange(4.0, requires_grad=True)`
print(x.grad)  # 默认None

# 计算 y
y = 2*torch.dot(x,x)  #tensor(28., grad_fn=<MulBackward0>)
# 通过反向传播自动计算 y 关于 x 每个分量的梯度
y.backward()
print(x.grad)  #tensor([ 0.,  4.,  8., 12.])
# 导数的计算4x， tensor([ 0.,  1.,  2., 3.]) -----> tensor([ 0.,  4.,  8., 12.])


y = x.sum()  #tensor(6., grad_fn=<MulBackward0>)
# 在默认情况下，PyTorch会累积梯度，我们需要在backwa前清除之前的值
x.grad.zero_()
print(x.grad)  #tensor([0., 0., 0., 0.])
y.backward()
print(x.grad)  #tensor([1., 1., 1., 1.])


#2.非标量变量的反向传播
print("============================")
x.grad.zero_()
y = x*x  #tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
# y.backward()  #error，非标量不能反向传播
y.sum().backward()  #等价于y.backward(torch.ones(len(x)))
print(x.grad)  #tensor([0., 2., 4., 6.])
# 导数的计算2x， tensor([ 0.,  1.,  2., 3.]) -----> tensor([ 0.,  2.,  4., 6.])

x.grad.zero_()
y = x*x*x
y.backward(torch.ones(len(x)))
print(x.grad)  #tensor([ 0.,  3., 12., 27.])
# 导数的计算3x*x， tensor([ 0.,  1.,  2., 3.]) -----> tensor([ 0.,  3.,  12., 27.])

#3.分离计算
print("==========================")
x.grad.zero_()
y = x * x
u = y.detach()  # 返回一个新的张量，但其不在计算图中
z = u * x   # 将 u 看作常数，则 x.grad = u

z.sum().backward()
print(x.grad == u)  #tensor([True, True, True, True])

x.grad.zero_()
y.sum().backward()
print(x.grad)  #tensor([0., 2., 4., 6.])

#4.python 控制流
print("==============================")
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
