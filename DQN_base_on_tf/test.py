#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: test.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/18 17:55:04
############################

import sys
sys.path.append(r'N:\reinforcement_learning\Qlearning\ql_maze_myself')
from ql_maze_class import Maze_m
import numpy as np
import pandas as pd

a = np.arange(12).reshape((3,4))
a = 1 + np.max(a, axis=1)
print(a)

#env = Maze_m()
#
#state = env.reset()
#a = np.array(state)
#
#t = np.hstack((a, [1,2], a))
##print(t)
##print(t.shape)
#
#memo = pd.DataFrame(np.zeros((200, 6)))
##print(memo)
#memo.iloc[0,:] = t
##print(memo.shape)
#
#temp = memo.iloc[0,:]
##print(temp)
##print(type(temp))
##print(temp[-2:].shape)
##print(temp[:2].shape)
#bat = memo.iloc[[0,3,4],:]
#print(bat.shape)
#print(bat)
#print(bat.iloc[:, 2].astype(int))
#
#
