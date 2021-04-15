#!/usr/bin/python3
#-*- coding:utf-8 -*-
############################
# File Name: test.py
# Author: taowang
# mail: 907993189@qq.com
# Create Time: 2021/4/15 18:46:58
############################

# copy from 
# http://blog.nodetopo.com/2020/03/23/tensorflow2%e5%ad%a6%e4%b9%a0%e5%9b%9b/
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# load mnist dataset
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data(path='mnist.npz')
# Normalize
train_x, test_x = train_x / 255.0, test_x / 255.0
train_x, test_x = train_x[:, :, :, np.newaxis], test_x[:, :, :, np.newaxis]

# show a picture of number
# number = train_x[0]
# plt.figure()
# plt.title('handwritten numberal')
# plt.imshow(number)
# plt.show()

model = keras.models.Sequential([
    keras.layers.Conv2D(28, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
])
# model.summary()
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y))

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
print('loss', test_loss, '\naccuracy: ',test_acc)
