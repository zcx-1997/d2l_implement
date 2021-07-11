# -*- coding: utf-8 -*-

"""
@Time : 2021/7/9
@Author : Lenovo
@File : 04-微分
@Description : 
"""

import numpy as np
from IPython import display
from d2l import torch as d2l
from matplotlib import pyplot as plt
from toolFunctions.visualFunctions import plot


# 1.导数
def f(x):
    ''' 函数 y = 3(x**2)-4x '''
    return 3 * x ** 2 - 4 * x

def numeral_lim(f,x,h):
    return (f(x+h)-f(x)) / h

h =0.1
for i in range(5):
    print("h ={:.5f}, numeral_lim={:.5f}".format(h,numeral_lim(f,1,h)))
    h*=0.1
'''
h =0.10000, numeral_lim=2.30000
h =0.01000, numeral_lim=2.03000
h =0.00100, numeral_lim=2.00300
h =0.00010, numeral_lim=2.00030
h =0.00001, numeral_lim=2.00003
'''

#绘制函数 f(x)及其在 x=1 处的切线 y=2x-3, 系数2是切线的斜率（在x=1处的导数）
x = np.arange(0,3,0.1)
plt.figure(figsize=(7, 5))
plt.xlabel('x')
plt.ylabel('f(x)')
p1, = plt.plot(x, f(x))
p2, = plt.plot(x, 2 * x - 3)
plt.legend([p1,p2],['f(x)','Tangent line(x=1)'])
plt.grid()
plt.show()