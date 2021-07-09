#!/usr/bin/python3
"""
    Time    : 2021/4/16 15:09
    Author  : 春晓
    Software: PyCharm
"""
import torch

# 1. arange创建数字列表
x = torch.arange(12)
print(x)
#tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 2. 张量常用属性和方法
print(x.shape)  #torch.Size([12])
#reshape改变张量形状
x = x.reshape(3,4)
print(x.shape)  #torch.Size([3, 4])

print(type(x))  #<class 'torch.Tensor'>
print(x.size())  #torch.Size([3, 4])
print(x.numel())  #12
print(x.dtype)  #torch.int64
print(len(x))  #12

#改变维度
x = x.reshape(2,2,3)
print(x.shape)  #torch.Size([2, 2, 3])
#自动计算维度
x = x.reshape(-1, 4)
print(x.shape)  #torch.Size([3, 4])

# 3.张量的常用创建方法
x1 = torch.zeros((2, 3))
print(x1)
x2 = torch.ones((2,3))
print(x2)
x3 = torch.rand((2,3))  #0~1
print(x3)
x4 = torch.randn((2,3))  #-1~1,正态分布
print(x4)
x5 = torch.tensor([[0,1,2],[3,4,5]])  #直接创建（从列表中创建）
print(x5)

# 4. 张量运算
#基本运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x**y, x//y)
#幂运算
a = torch.exp(torch.tensor(1))
print(a)  #tensor(2.7183)
#对数运算
b = torch.log(a)
print(b)  #tensor(1.)

#连接
a = torch.arange(6).reshape(2,3)
print(a)
'''
tensor([[0, 1, 2],
        [3, 4, 5]])
'''
b = torch.tensor([[1.,1],[1,1]])
print(b)
'''
tensor([[1., 1.],
        [1., 1.]])
'''
c = torch.cat((a,b),dim=1)
print(c)
'''
tensor([[0., 1., 2., 1., 1.],
        [3., 4., 5., 1., 1.]])
'''

#求和
x = torch.arange(5)
y = x.sum()
print(y)  #tensor(10)


# 5.广播机制
a = torch.arange(3).reshape((3, 1))
print(a)
'''
tensor([[0],
        [1],
        [2]])
'''
b = torch.arange(2).reshape((1, 2))
print(b)
'''
tensor([[0, 1]])
'''
c = a+b
print(c)
'''
tensor([[0, 1],
        [1, 2],
        [2, 3]])
'''


# 6.节省内存
before = id(a)
a = a + b  #重新分配内存
print(id(a) == before)  #False

before = id(a)
a += b  # 或 a[:] = a+b #不重新分配内存
print(id(a) == before)  #True


# 7.转换为其他 python对象
y1 = x.numpy()
print(type(y1))  #<class 'numpy.ndarray'>

y2 = torch.tensor(y1)
print(type(y2))  #<class 'torch.Tensor'>

#将大小为 1的张量转换为 python标量
a = torch.tensor([3.5])
print(a)  #tensor([3.5000])
print(a.item())  #3.5
print(float(a))  #3.5