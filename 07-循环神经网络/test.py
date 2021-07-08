#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/21 10:11
    Author  : 春晓
    Software: PyCharm
"""


import re
import collections

# def read_book():
#     with open('book1.txt','r') as f:
#         lines = f.readlines()
#         return lines
#
# lines = read_book()

with open('book1.txt','r') as f:
    lines = f.readlines()
    print(lines)
    # for line in lines:
    r1 = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    print(r1)

    r2 = [line.split() for line in r1]
    print(r2)

    r3 = [list(line) for line in r1]
    print(r3)

    tokens = [token for line in r3 for token in line]
    r4 = collections.Counter(tokens)
    print(r4)

unk=['unk']
blank = []
unk = unk +blank
print(unk)

import random
list1 = [0,5,10,15,20,25]
random.shuffle(list1)
print(list1)
# print(list2)