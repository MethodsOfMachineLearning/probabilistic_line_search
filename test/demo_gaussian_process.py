# -*- coding: utf-8 -*-
"""
Demo for Gaussian process functionality in probls.gaussian_process.

Created on Thu Nov 17 16:58:40 2016

@author: lballes
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import time

from probls import gaussian_process


# Specify noise levels and observations
fvar, dfvar = 3e-1, 1e-2
observations = [(0., 0., -1.), (1., -0.5, -0.9), (2., -0.9, 0.7)]

# Add observations to GP, compute posterior mean and variance
gp = gaussian_process.ProbLSGaussianProcess()
for obs in observations:
  gp.add(*obs, fvar=fvar, dfvar=dfvar)
beg = time.time()
gp.update()
print "gp.update() took", (time.time()-beg)*10**6, "microseconds"

tt = np.arange(-0.1, 4.0, 0.01)


fig, (a1, a2, a3) = plt.subplots(3, 1)
gp.visualize_f(a1)
gp.visualize_df(a2)
gp.visualize_ei(a3)

# Find the minima and add them to the plot
minima = gp.find_dmu_equal(0.2)
a1.plot(minima, [gp.mu(m) for m in minima], 'D')
a2.plot(minima, [gp.dmu(m) for m in minima], 'D')