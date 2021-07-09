#!/usr/bin/python3
"""
    Time    : 2021/4/23 17:18
    Author  : 春晓
    Software: PyCharm
"""

import torch

#1.标量
x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x+y,x*y,x/y,x**y)
#tensor([5.]) tensor([6.]) tensor([1.5000]) tensor([9.])


'''
a = torch.arange(12).reshape(3,4)
print(a)
b = a.T
print(b)

x = torch.arange(6).reshape(2,3)
y = x.clone()+1  # 通过分配新内存，将x的一个副本分配给y
print(x)
print(y)

# x = x.reshape(2,1,3)
# y = y.reshape(2,1,3)


z = torch.mul(x,y) # 或 z = x*y
print(z)


print(x.shape,y.T.shape)
z = torch.mm(x,y.T) # 或 z = x@y.T
print(z)

z1 = torch.matmul(x,y.T)
print(z1)



a = torch.arange(4)
b = torch.ones_like(a)
print(a)
print(b)
print(torch.dot(a,b))

print(x)
print(x.shape,x.sum())

print(x.shape)
print(x.sum(axis=0))
print(x.sum(axis=1))

sum_0 = x.sum(axis=0,keepdims=True)
sum_1 = x.sum(axis=1,keepdims=True)
print(sum_0,sum_0.shape)
print(sum_1,sum_1.shape)


u = torch.tensor([3.0, -4.0])
print("L1:",torch.abs(u).sum())
print("L2",torch.norm(u))
print("F:",torch.norm(torch.ones((4,9))))
'''