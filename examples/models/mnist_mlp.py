# -*- coding: utf-8 -*-
"""
TensorFlow MNIST MLP model.
"""

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.05, shape=shape)
  return tf.Variable(initial)

def set_up_model():
  tf.reset_default_graph()
  X = tf.placeholder(tf.float32, shape=[None, 784])
  y = tf.placeholder(tf.float32, shape=[None, 10])
  W_fc1 = weight_variable([784, 800])
  b_fc1 = bias_variable([800])
  h_fc1 = tf.nn.sigmoid(tf.matmul(X, W_fc1) + b_fc1)
  W_fc2 = weight_variable([800, 10])
  b_fc2 = bias_variable([10])
  h_fc2 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
  losses = -tf.reduce_sum(y*tf.log(h_fc2), reduction_indices=[1])
  return losses, [X, y], [W_fc1, b_fc1, W_fc2, b_fc2]
