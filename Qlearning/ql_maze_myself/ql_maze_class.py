#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: ql_maze_class.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/9 16:27:41
############################

import numpy as np

class Maze_m: # only cube n*n
    def __init__(self, martix=None):
        if martix == None:
            self.martix = np.zeros((4,4))
            self.martix[1,2] = self.martix[2,1] = -1
            self.martix[2,2] = 1
        else:
            self.martix = martix
        self.action_list = ['up','down', 'right', 'left']
        self.action_num = len(self.action_list)
        self.action_op = np.array([[-1,0],[1,0],[0,1],[0,-1]])
        self.posnow = np.array([0,0])
        self.maxl = self.martix.shape[1]

    def reset(self, martix=None):
        if martix == None:
            self.martix = np.zeros((4,4))
            self.martix[1,2] = self.martix[2,1] = -1
            self.martix[2,2] = 1
        else:
            self.martix = martix
        self.posnow = np.array([0,0])
        return [self.posnow[0],self.posnow[1]]

    def step(self, action):
        tmpindex = self.action_list.index(action)
        self.posnow[0]+=self.action_op[tmpindex,0]
        self.posnow[1]+=self.action_op[tmpindex,1]
        self.posnow[self.posnow<0] = 0
        self.posnow[self.posnow>=self.maxl] = self.maxl-1
        reward = self.martix[self.posnow[0],self.posnow[1]]
        if reward == 1 or reward == -1:
            done = True
        else:
            done = False
        return [self.posnow[0],self.posnow[1]], reward, done

    def print_maze(self):
        a = self.martix.copy()
        a[self.posnow[0],self.posnow[1]] = 9
        print(a)
    
    def get_pos(self):
        return [self.posnow[0],self.posnow[1]]

#tmp = Maze_m()
#tmp.print_maze()
#tmp.step('down')
#tmp.print_maze()
