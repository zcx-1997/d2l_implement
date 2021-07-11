#!/usr/bin/python3
"""
    Time    : 2021/4/23 17:18
    Author  : 春晓
    Software: PyCharm
"""
import numpy
import torch

#1.标量
x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x+y,x*y,x/y,x**y)
#tensor([5.]) tensor([6.]) tensor([1.5000]) tensor([9.])


#2.向量
x = torch.arange(4)
print(x)  #tensor([0, 1, 2, 3])
print(x[1])  #tensor(1)
print(len(x))  #4
print(x.shape)  #torch.Size([4])


#3.矩阵
a = torch.arange(12).reshape(3,4)
print(a)
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
'''
b = a.T
print(b)
'''
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
'''


#4.张量
x = torch.arange(24).reshape(2,3,4)
print(x)
'''
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
'''


#5.求和、平均值
a = torch.arange(6,dtype=torch.float32).reshape(2,3)
print(a)
'''
tensor([[0., 1., 2.],
        [3., 4., 5.]])
'''

sum_a = a.sum()
print(sum_a)  #tensor(15.)


sum_axis0 = a.sum(axis=0)
print(sum_axis0)  #tensor([3., 5., 7.])

sum_axis1 = a.sum(axis=1)
print(sum_axis1)  #tensor([ 3., 12.])

#平均值
mean_a = a.mean()
print(mean_a)  #tensor(2.5000)
print(a.sum()/a.numel())  #tensor(2.5000)

mean_axis0 = a.mean(axis=0)
print(mean_axis0)  #tensor([1.5000, 2.5000, 3.5000])

mean_axis = a.sum(axis=0) / a.shape[0]
print(mean_axis)  #tensor([1.5000, 2.5000, 3.5000])

#保存维度不变
sum = a.sum(axis=1,keepdims=True)
print(sum)
'''
tensor([[ 3.],
        [12.]])
'''


#6.向量点积
a = torch.arange(4)
b = torch.ones_like(a)
print(a)  #tensor([0, 1, 2, 3])
print(b)  #tensor([1, 1, 1, 1])
print(torch.dot(a,b))  #tensor(6)
print(torch.sum(a*b))  #tensor(6)


#7.矩阵-向量积
a = torch.arange(6).reshape(2,3)
x = torch.arange(3)
print(a)
'''
tensor([[0, 1, 2],
        [3, 4, 5]])
'''
print(x)  #tensor([0, 1, 2])
print(numpy.dot(a,x))  #[ 5 14]
print(torch.mv(a,x))  #tensor([ 5, 14])

#8.矩阵乘法和哈达玛积
x = torch.arange(6).reshape(2,3)
y = x.clone()+1  # 通过分配新内存，将x的一个副本分配给y
y = y.T
print(x)
'''
tensor([[0, 1, 2],
        [3, 4, 5]])
'''
print(y)
'''
tensor([[1, 4],
        [2, 5],
        [3, 6]])
'''

z = torch.mm(x,y)  #或torch.matmul(x,y) 或z = x@y
# torch.matmul可以进行张量乘法, 输入可以是高维. x@y只针对矩阵（二维张量）
print(z)
'''
tensor([[ 8, 17],
        [26, 62]])
'''

#哈达玛积，形状要相同
z = torch.mul(x,y.T) # 或 z = x*y
print(z)
'''
tensor([[ 0,  2,  6],
        [12, 20, 30]])
'''


#9.范数
u = torch.tensor([3.0, -4.0])
print("L1:",torch.abs(u).sum())  #L1: tensor(7.)
print("L2",torch.norm(u))  #L2 tensor(5.)
print("F:",torch.norm(torch.ones((4,9))))  #F: tensor(6.)