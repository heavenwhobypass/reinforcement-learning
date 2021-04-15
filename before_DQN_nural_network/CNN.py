#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: CNN.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/14 20:33:30
############################
# 输入集改了一下源
# loss function 改成平方
# 就熟悉一下tensorflow 然后跑出来是什么东西我也不清楚
# 目前需要补一补理论知识 以后回来看看这堆代码跑出来为什么不明所以@TODO

# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow.keras as keras
import numpy as np

# number 1 to 10 data
(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
x = x.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))
ynew = np.zeros((60000, 10))
for i in range(60000):
    ynew[i, y[i]] = 1
y = ynew
ynew = np.zeros((10000,10))
for i in range(10000):
    ynew[i, y_test[i]] = 1
y_test = ynew
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32]) # pathc 5x5 in size 1, outsize 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                     # output size 14x14x32
## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64]) # pathc 5x5 in size 32, outsize 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                     # output size 7x7x64
## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean( tf.reduce_sum(tf.square(tf.subtract(ys, prediction)), reduction_indices=[1]))
#tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    batch_xs, batch_ys = x[i*100:i*100+100], y[i*100:i*100+100]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 20 == 0:
        #print(sess.run(W_conv1[0]))
        print(compute_accuracy(
            x_test[:1000], y_test[:1000]))

