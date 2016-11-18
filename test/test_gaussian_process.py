# -*- coding: utf-8 -*-
"""
Test for Gaussian process implementation in probls.gaussian_process

Created on Fri Jul  1 09:51:00 2016

@author: lballes
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np


from probls import gaussian_process


class TestSolveQuadraticPolynomial(unittest.TestCase):
  """Test the ``quadratic_polynomial_solve`` function of
  ``probls.gaussian_process`` with a few hand-computed polynomials."""
  
  def setUp(self):
    self.solve = gaussian_process.quadratic_polynomial_solve
  
  def runTest(self):
    self.assertListEqual(self.solve(1., 0., 0., -3.5), [])
    self.assertListEqual(self.solve(1., 0., 0., 0.), [])
    self.assertListEqual(self.solve(2., -4., 0., 0.), [2.])
    self.assertListEqual(self.solve(1., 3., -2., -4.), [-1.])
    self.assertListEqual(self.solve(2., 0.5, 4., -8.0), [])


class TestKernelFunctions(unittest.TestCase):
  
  def setUp(self):
    self.gp = gaussian_process.ProbLSGaussianProcess()
  
  def runTest(self):
    
    # Test kernel function with hand-computed values
    self.assertEqual(self.gp.k(3.5, 1.), 11.**3/3. + 0.5*2.5*11.**2)
    self.assertEqual(self.gp.k(2., 3.), 12.**3/3.+.5*12.**2)    
    self.assertEqual(self.gp.dkd(1., 2.0), 11.)
    self.assertEqual(self.gp.dkd(-2., -1.), 8.)
    
    # Test if one-to-one computations give the same result as  one-to-many
    # computations
    t, T = np.random.rand(), np.random.rand(10)
    for fun in [self.gp.k, self.gp.kd, self.gp.dkd, self.gp.d2k, self.gp.d2kd, self.gp.d3k]:
      res = fun(t, T)
      for i, tt in enumerate(T):
        self.assertEqual(fun(t, tt), res[i])


class TestNoiseFree(unittest.TestCase):
  """Test whether posterior mean equals observations in the noise-free case."""
  
  def setUp(self):
    self.gp = gaussian_process.ProbLSGaussianProcess()
  
  def runTest(self):
    ts, fs, dfs = np.random.randn(10), np.random.randn(10), np.random.randn(10)
    for i in range(10):
      self.gp.add(ts[i], fs[i], dfs[i])
    self.gp.update()
    for i in range(10):
      t, f, df = ts[i], fs[i], dfs[i]
      self.assertLess(self.gp.V(t), 1e-9)
      self.assertAlmostEqual(self.gp.mu(t), f, places=3)
      self.assertAlmostEqual(self.gp.dmu(t), df, places=3)


if __name__ == "__main__":
  unittest.main()