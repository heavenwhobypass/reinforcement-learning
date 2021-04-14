#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: calssification_minist.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/13 19:11:42
############################

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow.keras as keras
import numpy as np

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
#per
x = x.reshape(-1,784)
x_test = x_test.reshape(-1,784)
ynew = np.zeros((60000, 10))
for i in range(60000):
    ynew[i, y[i]] = 1
y = ynew
ynew = np.zeros((10000,10))
for i in range(10000):
    ynew[i, y_test[i]] = 1
y_test = ynew

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None :
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):

    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    print(y_pre[1,:], v_ys[1,:])
    correct_preiction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preiction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

# define placeholder
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

#add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean( tf.reduce_sum(tf.square(tf.subtract(ys, prediction)), reduction_indices=[1]))
#-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])

train_step = tf.train.GradientDescentOptimizer(0.51).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x, ys:y})
        if i%50 ==0 :
            print(compute_accuracy(x_test, y_test))
