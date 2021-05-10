#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: DQN_torch.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/5/8 14:10:42
############################

import torch
import pandas as pd
import numpy as np
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class DQN:
    def __init__(self,
            n_actions,
            n_features,
            learning_rate = 0.01,
            discount = 0.9,
            e_greedy=0.1,
            e_greedy_decrement=None,
            replace_target_iter = 300,
            memory_size = 500,
            batch_size=32,#32,
            ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon_min = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.epsilon_decrement = e_greedy_decrement
        self.epsilon = 1 if e_greedy_decrement is not None else self.epsilon_min
        #self.epsilon = e_greedy
        self.batch_size= batch_size
        # learn step
        self.learn_step_counter = 0
        # [s, a , r, s_ ]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features * 2 + 2), dtype=np.float32))
        # build net
        self._build_net()
        #loss
        self.loss_history = []
        #
        self.memory_counter = 0
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.now_q_NN.parameters(), self.lr)

    def _build_net(self):
        self.now_q_NN = Q_NN().to(device)
        self.tar_q_NN = Q_NN().to(device)
        self.replace_target_parameter()
        pass

    # 复制参数到 target网络
    def replace_target_parameter(self):
        # 这两个函数可以看下 1.0.0 torch 文档
        self.tar_q_NN.load_state_dict(self.now_q_NN.state_dict())
        pass

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1
        pass

    def choose_action(self, state, print_flag):
        # to tensor
        state = torch.from_numpy(state).to(device)
        #print(state)

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                actions_value = self.now_q_NN(state)
                if print_flag:
                    print(actions_value)
            action = np.argmax(actions_value.cpu().numpy())
        return action
        pass

    def learn(self):
        # 先看要不要 换参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_parameter()
            print("\n target_parameters_repalced\n")

        # sample 
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory.iloc[sample_index, :]
        batch_memory = batch_memory.to_numpy() # pandsDF 变 numpy
        #batch_memory = torch.from_numpy(batch_memory.to_numpy()) # pandsDF 变 numpy

        b_s = torch.FloatTensor(batch_memory[:, :self.n_features]).to(device)
        b_a = torch.LongTensor(batch_memory[:, self.n_features:self.n_features + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(batch_memory[:, self.n_features + 1:self.n_features + 2]).to(device)
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:]).to(device)

        #print(b_a.shape)
        q_eval = self.now_q_NN(b_s).gather(1, b_a)
        q_next = self.tar_q_NN(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss_fn( q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        sum_loss = loss.item()
        self.learn_step_counter += 1
        # 获得 q_target 也就是 所谓的 label
        ##q_target = None
        ##with torch.no_grad():
        ##    for i in range(self.batch_size):
        ##        s = batch_memory[i,:self.n_features].to(device)
        ##        s_ = batch_memory[i,-self.n_features:].to(device)
        ##        q_eval = self.now_q_NN(s)
        ##        # 就是如果是s_终结state就不加了
        ##        if is_terminal(s_):
        ##            q_next = torch.tensor([0])
        ##        else:
        ##            q_next = self.tar_q_NN(s_)
        ##        #
        ##        q_target_tmp = q_eval.clone()
        ##        eval_act_index = batch_memory[i, self.n_features] #0 1 2 第二位是aciton
        ##        reward = batch_memory[i, self.n_features+1]
        ##        # TODO tensor 可以进行numpy操作吗？
        ##        # index 必须转成 int
        ##        q_target_tmp[eval_act_index.type(torch.int64)] = reward + self.gamma * torch.max(q_next)
        ##        #
        ##        if q_target is None:
        ##            q_target = torch.unsqueeze(q_target_tmp, dim=0)
        ##        else:
        ##            #q_target = np.vstack([q_target,q_target_tmp])
        ##            q_target = torch.cat([q_target, torch.unsqueeze(q_target_tmp, dim=0)], 0)
        ##        #print(q_eval)
        ##        #print(q_next)
        ##        #print(q_target_tmp)
        ##        #print(eval_act_index.type(torch.int64))
        ##        #print(torch.max(q_next))
        ##        #print(q_target.shape)
        ##        #print(q_target)
        ##    #print("-------end sample")

        ##        # 正式 的 训练
        ##
        ##sum_loss = 0
        ##for i in range(self.batch_size):
        ##    X = batch_memory[i, :self.n_features].to(device)
        ##    y = q_target[i, :].to(device)
        ##    # compute loss
        ##    pred = self.now_q_NN(X)
#       ##     print(pred.shape)
#       ##     print(y.shape)
#       ##     pred = torch.unsqueeze(pred, 0)
#       ##     y = torch.unsqueeze(y, 0)
#       ##     print(pred.shape)
#       ##     print(y.shape)
#       ##     print(pred)
#       ##     print(y)
        ##    #print(pred)
        ##    #print(y)
        ##    loss = torch.sum(torch.pow((y - pred), 2))
        ##    #loss = self.loss_fn(pred,y)
        ##    # back propagation
        ##    self.optimizer.zero_grad()
        ##    loss.backward(loss.clone().detach())
        ##    self.optimizer.step()
        ##    #record loss
        ##    #print(loss.item())
        ##    sum_loss += loss.item()
        ##    self.learn_step_counter += 1
        self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_min else self.epsilon_min
        self.loss_history.append(sum_loss)
#        input()
        pass

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.show()
        pass

    # 读之前是不是已经训练过网络了
    def is_exist_pre_version(self):
        try:
            self.now_q_NN.load_state_dict(torch.load("now_q_NN.pth"))
        except BaseException:
            print("have no file name now_q_NN.pth")
        finally:
            print("successfully get now_q weight")
        try:
            self.tar_q_NN.load_state_dict(torch.load("tar_q_NN.pth"))
        except BaseException:
            print("have no file name tar_q_NN.pth")
        finally:
            print("successfully get tar_q weight")

    def save_net_weight(self):
        torch.save(self.now_q_NN.state_dict(), "now_q_NN.pth")
        torch.save(self.tar_q_NN.state_dict(), "tar_q_NN.pth")
        print("save model to now_q_NN.pth and tar_q_NN.pth")
        pass

    def save_loss(self):
        # @TODO
        pass


class Q_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                #nn.ReLU(),
                )

    def forward(self, x):
        action_value = self.linear_relu_stack(x)
        return action_value

#model = DQN(4, 2)
#print(model.choose_action(np.array([0.,0.], dtype=np.float32)))

# TODO  根据 maze要改的
def is_terminal(s_):
    if s_[0] == 4 and s_[1] == 4:
        return True
    elif s_[0] == 1 and s_[1] == 3:
        return True
    elif s_[0] == 3 and s_[1] == 1:
        return True
    return False


