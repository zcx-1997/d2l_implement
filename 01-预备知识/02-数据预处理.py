#!/usr/bin/python3
"""
    Time    : 2021/4/23 16:28
    Author  : 春晓
    Software: PyCharm
"""
import os
import torch
import pandas as pd

# 创建并读取数据集
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 处理缺失值
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs,dummy_na=True)
print(inputs)

x,y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)