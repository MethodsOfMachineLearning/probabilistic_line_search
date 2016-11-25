# -*- coding: utf-8 -*-
"""
Tests for gradient moment computation in
probls.tensorflow_interface.gradient_moment

Created on Wed Nov 23 17:09:34 2016

@author: Lukas Balles [lballes@tuebingen.mpg.de]
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import unittest
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from probls.tensorflow_interface import gradient_moment as gm

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.05, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class TestGradientMomentFullyConnected(unittest.TestCase):
  """Test."""
  
  def setUp(self):    
    # Set up model
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    W_fc1 = weight_variable([784, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    h_fc2 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    losses = -tf.reduce_sum(y*tf.log(h_fc2), reduction_indices=[1])
    
    self.loss = tf.reduce_mean(losses)
    self.batch_size = tf.cast(tf.gather(tf.shape(losses), 0), tf.float32)
    self.var_list = [W_fc1, b_fc1, W_fc2, b_fc2]
    self.X = X
    self.y = y
    
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())
    
    self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
  def runTest(self):
    grads, grad_moms = gm.grads_and_grad_moms(self.loss, self.batch_size,
                                              self.var_list)
    # Check shapes
    for v, g, mom in zip(self.var_list, grads, grad_moms):
      self.assertEqual(v.get_shape(), g.get_shape())
      self.assertEqual(v.get_shape(), mom.get_shape())
    
    # Check against manual computation of moment
    m = 10
    batch = self.mnist.train.next_batch(m)
    Xb, yb = batch[0], batch[1]
    indiv_grads = []
    for i in range(m):
      gs = self.sess.run(grads, feed_dict={self.X: Xb[[i],:], self.y: yb[[i],:]})
      indiv_grads.append(gs) 
    indiv_grads_arr = [np.stack([indiv_grads[i][j] for i in range(m)], axis=0) for j in range(len(self.var_list))]
    grads_manual = [np.mean(gs_var, axis=0) for gs_var in indiv_grads_arr]
    grad_moms_manual = [np.mean(gs_var**2, axis=0) for gs_var in indiv_grads_arr]
    grads_impl, grad_moms_impl = self.sess.run([grads, grad_moms], feed_dict={self.X: Xb, self.y: yb})
    for grm, gri in zip(grads_manual, grads_impl):
        self.assertTrue(np.allclose(grm, gri, rtol=1e-4))
    for gmm, gmi in zip(grad_moms_manual, grad_moms_impl):
        self.assertTrue(np.allclose(gmm, gmi, rtol=1e-4))

if __name__ == "__main__":
  unittest.main()