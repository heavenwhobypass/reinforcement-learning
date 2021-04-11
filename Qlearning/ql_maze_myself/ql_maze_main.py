#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: ql_maze_main.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/9 17:34:27
############################

from ql_maze_class import Maze_m
from ql_maze_brain import Qltab

import os
import time

episodes = 500
countpostive = 0
countlist = []

maze_test = Maze_m()
RL = Qltab()

def ql_main():
    for episode in range(episodes):
        state = maze_test.get_pos()
        #maze_test.print_maze()
        if len(countlist) != 0:
            print(len(countlist), countlist[-1])
        count = 0
        while True:
            #choose action [str]
            action = RL.choose_action(''.join(str(i) for i in state))
            #do action  state= [a,b]
            state_next, reward, done = maze_test.step(action)
            count += 1
            if reward > 0:
                global countpostive
                countpostive = countpostive  + 1

            # print
           # time.sleep(0.1)
           # os.system("cls")
           # print(action)
           # maze_test.print_maze()
            # print

            #learn
            RL.learn(''.join(str(i) for i in state), action, reward, ''.join(str(i) for i in state_next))
            #update
            state = state_next
            #donw break
            if done: # + yes  - no
                countlist.append(count*reward)
                break
        state = maze_test.reset()
    RL.print_qtable()
    print(countlist)
    print(countpostive)


ql_main()

#test
#tmp = ''.join(str(i) for i in maze_test.get_pos())
#action = RL.choose_action(tmp)
#RL.print_qtable()
#print(action)
#a,b,c= maze_test.step(action)
#
#RL.learn(''.join(str(i) for i in tmp), action, b, ''.join(str(i) for i in a))
#RL.print_qtable()
