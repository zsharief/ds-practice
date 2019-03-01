# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:03:54 2019

@author: Zahid
"""

# https://machinelearningmastery.com/introduction-python-deep-learning-library-tensorflow/

## nodes - perform computation and have zero or more inputs and outputs; data moves between nodes
## tensors - multi dimensional arrays of real values
## edges - define the flow of data; branching, looping and updates to state
## operation - named abstract computation which takes input attributes and produces output attributes

## computation with tensorflow ##
import tensorflow as tf
sess = tf.Session()
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))

## linear regression with tensorflow ##
## tensorflow separates the definition and declaration of the computation from the execution
## in the session and the calls to run.
import tensorflow as tf 
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

## minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

## initialize all tf variables before starting
init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()
## above step required for tf.Variables

## launch the graph
sess = tf.Session()
sess.run(init)

## fit the line
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
## learns best fit is W:[0.1], b:[0.3]

## close session to free up some resources
sess.close()