#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: sarsa_lambda_maze_brain.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/12 17:04:16
############################

import numpy as np
import pandas as pd

class Sarsa:
    def __init__(self, actions=['up','down', 'right', 'left'], learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.sarsa_table = pd.DataFrame(columns=actions, dtype=np.float64)
        #new
        #backward view, eligibility trace
        self.lambda_ = trace_decay
        self.eligibility_trace = self.sarsa_table.copy()

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
        error = sarsa_target - sarsa_predict
        
        # increase trace amount for visited state-aciton pair

        # method 1:
        #self.eligibility_trace.loc[state, action] += 1
        # method 2:
        self.eligibility_trace.loc[state, :] *= 0 # 可以减少对环路的学习
        self.eligibility_trace.loc[state, action] = 1

        # q update
        self.sarsa_table += self.lr * error * self.eligibility_trace

        #decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_

    def check_is_exist(self, state):
        if state not in self.sarsa_table.index:
            to_be_append = pd.Series([0]*len(self.actions), index = self.sarsa_table.columns, name = state)
            self.sarsa_table = self.sarsa_table.append(to_be_append)
            #also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def print_table(self):
        print(self.sarsa_table)
    
    def print_Etable(self):
        print("eligibility-table")
        print(self.eligibility_trace)
