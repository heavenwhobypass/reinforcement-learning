#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: sarsa_maze_brain.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/11 12:38:37
############################
import numpy as np
import pandas as pd

class Sarsa:
    def __init__(self, actions=['up','down', 'right', 'left'], learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.sarsa_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_is_exist(state)
        tmp = self.sarsa_table.loc[state, :]
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(tmp[tmp==tmp.max()].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, state_next, action_next):
        self.check_is_exist(state_next)
        sarsa_predict = self.sarsa_table.loc[state, action]
        if reward != 0:
            sarsa_target = reward
        else:
            sarsa_target = reward + self.gamma * self.sarsa_table.loc[state_next, action_next]
        #learn
        self.sarsa_table.loc[state, action] += self.lr * (sarsa_target - sarsa_predict)

    def check_is_exist(self, state):
        if state not in self.sarsa_table.index:
            self.sarsa_table = self.sarsa_table.append(
                    pd.Series(
                        [0]*len(self.actions),
                        index = self.sarsa_table.columns,
                        name = state,
                        ))

    def print_table(self):
        print(self.sarsa_table)
