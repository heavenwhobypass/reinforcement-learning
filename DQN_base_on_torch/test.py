#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: test.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/18 17:55:04
############################

import sys
import torch
sys.path.append(r'N:\reinforcement_learning\Qlearning\ql_maze_myself')
from ql_maze_class import Maze_m
import numpy as np
import pandas as pd


env = Maze_m()

state = env.reset()
a = np.array(state)

t = np.hstack((a, [1,2], a))
#print(t)
#print(t.shape)

memo = pd.DataFrame(np.zeros((200, 6)))
#print(memo)
memo.iloc[0,:] = t
#print(memo.shape)

temp = memo.iloc[0,:]
#print(temp)
#print(type(temp))
#print(temp[-2:].shape)
#print(temp[:2].shape)
bat = memo.iloc[[0,3,4],:]
print(bat.shape)
print(bat.to_numpy())
print(torch.from_numpy(bat.to_numpy()))
print(bat.iloc[:, 2].astype(int))

with torch.no_grad():
    tmp1 = torch.tensor([1,2])
    tmp2 = tmp1.clone()
    print(tmp2 )
    print(tmp1)
    tmp1[1] = 3
    print(tmp1)
    print(tmp2 )

qwe = [1,2]
asd = [3,4]
#qwe = np.vstack([qwe,asd])
#print(np.vstack([qwe,asd]))

#a = torch.tensor([1,1]).cuda()
#print(a.numpy())

# 实现 堆叠 
qwe = torch.tensor(qwe)
qwe = torch.unsqueeze(qwe, dim=0)
print(qwe.shape)
asd = torch.tensor(asd)
asd = torch.unsqueeze(asd, dim=0)
c = torch.cat([qwe, asd], 0)
print(c)
c = torch.cat([c, asd], 0)
print(c)
if c[0][0] == 1 and c[0][1] == 2:
    print("1")
else:
    print("not 1")
#c = None
#if c is None:
#    print("asdas")
