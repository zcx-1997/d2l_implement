# -*- coding: utf-8 -*-

"""
@Time : 2021/7/11
@Author : Lenovo
@File : 07-查阅文档
@Description : 
"""
import torch

#查找模块中的所有函数和类
print(dir(torch))

#查找特定函数和类的⽤法
print(help(torch.ones))