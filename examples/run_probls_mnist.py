# -*- coding: utf-8 -*-
"""
Run probabilistic line search on a MNIST example.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

from probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from probls.line_search import ProbLSOptimizer

#### Specify training specifics here ##########################################
#from models import mnist_2conv_2dense as model # Comment/uncomment to chose
from models import mnist_mlp as model           # the model to run 
num_steps = 4000
batch_size = 256
###############################################################################


# Set up model
losses, placeholders, variables = model.set_up_model()
X, y = placeholders

# Set up ProbLS optimizer
opt_interface = ProbLSOptimizerSGDInterface()
opt_interface.minimize(losses, variables)
sess = tf.Session()
opt_interface.register_session(sess)
sess.run(tf.initialize_all_variables())
opt_ls = ProbLSOptimizer(opt_interface, alpha0=1e-3, cW=0.3, c1=0.05,
    target_df=0.5, df_lo=-0.1, df_hi=1.1, expl_policy="linear", fpush=1.0,
    max_change_factor=10., max_steps=10, max_expl=10, max_dmu0=0.0)
batch = mnist.train.next_batch(batch_size)
opt_ls.prepare({X: batch[0], y: batch[1]})

# Run ProbLS
for i in range(num_steps):
  batch = mnist.train.next_batch(batch_size)
  print(opt_ls.proceed({X: batch[0], y: batch[1]}))