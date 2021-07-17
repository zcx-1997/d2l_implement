"""
    Time    : 2021/5/19 20:30
    Author  : 春晓
    Software: PyCharm
"""
import torch

x, w_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
h, w_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))

logits = torch.matmul(x, w_xh) + torch.matmul(h, w_hh)
print(logits)
'''
tensor([[-3.3239,  0.2946, -1.8694, -1.1380],
        [ 3.3800, -1.5331,  2.4712, -1.1515],
        [ 2.3279,  0.6343,  0.8435,  1.0886]])
'''

logits2 = torch.matmul(torch.cat((x, h), 1), torch.cat((w_xh, w_hh), 0))
print(logits2)
'''
tensor([[ 1.0026, -0.3073,  3.9255, -1.2311],
        [ 0.3160, -0.4164,  1.1474, -4.1001],
        [ 0.1424, -0.6117, -0.8720, -3.6993]])
'''
