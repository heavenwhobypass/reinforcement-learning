#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: ql_main.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/9 15:02:00
############################

from ql_maze_env import Maze
from ql_brain import QLearningTable

def update():
    for episode in range(100):
        #init
        observation = env.reset()
        while True:
            # fresh env
            env.render()
            #choose action
            action = RL.choose_action(str(observation))
            #
            observation_, reward, done = env.step(action)
            #
            RL.learn(str(observation), action, reward, str(observation_))
            #
            observation = observation_
            if done:
                break
    print('game over')
    env.destroy()
if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(100,update) # after tkinter
    env.mainloop()
    print(RL.q_table)


