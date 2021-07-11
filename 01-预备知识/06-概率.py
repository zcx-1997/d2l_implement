# -*- coding: utf-8 -*-

"""
@Time : 2021/7/11
@Author : Lenovo
@File : 06-概率
@Description : 
"""

import torch
from torch.distributions import multinomial
fair_probs = torch.ones([6]) / 6
result = multinomial.Multinomial(1,fair_probs).sample()
print(result)  #tensor([0., 0., 0., 1., 0., 0.])

result = multinomial.Multinomial(10,fair_probs).sample()
print(result)  #tensor([0., 2., 2., 0., 4., 2.])

counts = multinomial.Multinomial(1000,fair_probs).sample()
result = counts / 1000
print(result)  #tensor([0.1580, 0.1760, 0.1550, 0.1510, 0.1880, 0.1720])
