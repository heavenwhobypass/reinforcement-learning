#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: ql_maze_brain.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/9 17:34:08
############################

import numpy as np
import pandas as pd

class Qltab:
    def __init__(self, actions=['up','down', 'right', 'left'], learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_tab = pd.DataFrame(columns = actions, dtype=np.float64)
    def choose_action(self, state):
        self.check_is_exist(state)
        tmp = self.q_tab.loc[state,:]
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(tmp[tmp==tmp.max()].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, state_next):
        self.check_is_exist(state_next)
        q_predict = self.q_tab.loc[state,action]
        if reward == 1 or reward == -1: # 完成任务有 1 的反馈  -1 失败反馈
            q_target = reward
        else:
            q_target = reward + self.gamma * self.q_tab.loc[state_next,:].max()
        self.q_tab.loc[state,action] += self.lr * (q_target - q_predict) #****************

    def check_is_exist(self, state):
        if state not in self.q_tab.index:
            self.q_tab = self.q_tab.append(
                    pd.Series([0] * len(self.actions),
                        index = self.q_tab.columns,
                        name = state,
                        )
                    )

    def print_qtable(self):
        print(self.q_tab)

