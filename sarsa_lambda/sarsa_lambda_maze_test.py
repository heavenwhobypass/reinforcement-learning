#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: sarsa_lambda_maze_test.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/12 17:25:23
############################

#reward - [-1, 1, 0]
import sys
import os
import time

episodes = 500
countpostive = 0
countlist = []

#import maze class Maze_m
#__init__(martix=None)
#reset(martix) list[x, y]
#step(action) [x, y], reward, done
#print_maze()
#get_pos()   list[x, y]
sys.path.append(r'N:\reinforcement_learning\Qlearning\ql_maze_myself')
from ql_maze_class import Maze_m
from sarsa_lambda_maze_brain import Sarsa

def sarsa_process():
    for episode in range(episodes):
        state = env.reset()
        action = RL.choose_action(list2string(state))
        #env.print_maze()
        #RL.print_table()
        if len(countlist) != 0:
            print(len(countlist), countlist[-1])
        count = 0
        while True:
            #next step
            state_next, reward, done = env.step(action)

            #pirnt
            if reward > 0:
                global countpostive
                countpostive = countpostive  + 1
            count += 1

            #time.sleep(0.1)
            #os.system("cls")
            #print(action)
            #env.print_maze()
            #RL.print_table()
            #print
            #choose
            action_next = RL.choose_action(list2string(state_next))
            #learn
            RL.learn(list2string(state), action, reward, list2string(state_next), action_next)
            #update
            action = action_next
            state = state_next
            if done:
                countlist.append(count*reward)
                break
        #RL.print_Etable()
        state = env.reset()

def list2string(state):
    return ''.join(str(i) for i in state)


if __name__ == "__main__":
    #init maze
    env = Maze_m()
    RL = Sarsa(trace_decay=1)
    #sarsa rl
    sarsa_process()
    RL.print_table()
    print(countlist)
    print(countpostive)
