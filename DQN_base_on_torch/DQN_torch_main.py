#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: DQN_torch_main.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/5/8 14:12:17
############################

import sys
import os
import time
import numpy as np

sys.path.append(r'N:\reinforcement_learning\Qlearning\ql_maze_myself')
from ql_maze_class import Maze_m
from DQN_torch import DQN

episodes = 100
countpostive = 0
countlist = []
action_list = ['up','down', 'right', 'left']

def list2string(state):
    return ''.join(str(i) for i in state)
def DQN_main():
    step = 0
    for episode in range(episodes):
        print(f"episode {episode} \n")
        state = env.reset()

#record
        count = 0

        while True:
            # choose 
            p = False
            if episode>98:
                p=True
            #    os.system("cls")
            action_num = RL.choose_action(np.array(state, dtype=np.float32), p)
            action = action_list[action_num]
            # RL take action
            state_next, reward, done = env.step(action)
            #reward = reward * count

#record
            count += 1
            if reward > 0:
                global countpostive
                countpostive = countpostive  + 1
            if episode>98 :
                time.sleep(0.1)
                #os.system("cls")
                print(action)
                env.print_maze()
                print(reward)

            # store memory
            RL.store_transition(np.array(state), action_num, reward, np.array(state_next))
            # learn
            # 控制学习起始时间和频率
            # 先积累一些记忆再学习
            if (step>200) and (step%5==0):
                RL.learn()
            # state 
            state = state_next
            # stop
            if done:
                countlist.append(count*reward)#count*
                break
            step += 1
    print(countlist)
    print(countpostive)

if __name__ == "__main__":
    #init maze
    testa = np.array([[0,0,0,0,0],
                     [0,0,0,-1,0],
                     [0,0,0,0,0],
                     [0,-1,0,0,0],
                     [0,0,0,0,1],])
    env = Maze_m(testa)
    #init RL
    # env.action_list
    RL = DQN(4, 2, # xxx?
            learning_rate=0.01,
            discount=0.9,
            e_greedy=0.1,
            replace_target_iter=300,
            memory_size=2000,
            )
    # process
    DQN_main()
    # cost curve
    RL.plot_loss()
