# -*- coding: utf-8 -*-
"""
Run probabilistic line search on a CIFAR-10 example.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf

from probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from probls.line_search import ProbLSOptimizer

import cifar10

#### Specify training specifics here ##########################################
from models import cifar10_2conv_3dense as model
num_steps = 4000
batch_size = 256
###############################################################################


# Set up model
tf.reset_default_graph()
images, labels = cifar10.distorted_inputs(batch_size=batch_size)
losses, variables = model.set_up_model(images, labels)

# Set up ProbLS optimizer
opt_interface = ProbLSOptimizerSGDInterface()
opt_interface.minimize(losses, variables)
sess = tf.Session()
opt_interface.register_session(sess)
opt_ls = ProbLSOptimizer(opt_interface, alpha0=1e-3, cW=0.3, c1=0.05,
    target_df=0.5, df_lo=-0.1, df_hi=1.1, expl_policy="linear", fpush=1.0,
    max_change_factor=10., max_steps=10, max_expl=10, max_dmu0=0.0)

# Initialize variables and start queues
coord = tf.train.Coordinator()
sess.run(tf.initialize_all_variables())
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Run ProbLS
opt_ls.prepare()
for i in range(num_steps):
  print(opt_ls.proceed())

# Stop queues
coord.request_stop()
coord.join(threads)