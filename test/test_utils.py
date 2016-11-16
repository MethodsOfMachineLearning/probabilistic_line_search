# -*- coding: utf-8 -*-
"""
Tests for utility functions in probls.utils

Created on Wed Jul  6 16:12:54 2016

@author: lballes
"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np


from probls import utils
cdf = utils._cdf
bvnu = utils.unbounded_bivariate_normal_integral
bvn = utils.bounded_bivariate_normal_integral


class TestCDF(unittest.TestCase):
  
  def runTest(self):
    self.assertEqual(utils._cdf(0.), 0.5)
    self.assertAlmostEqual(cdf(1.), 0.8413, places=4)
    self.assertAlmostEqual(cdf(3.), 0.9987, places=4)
    self.assertAlmostEqual(cdf(-1.), 0.1587, places=4)
    self.assertAlmostEqual(cdf(-0.1), 0.4602, places=4)


class TestUnboundedIntegral(unittest.TestCase):
  
  def runTest(self):
    self.assertEqual(bvnu(0., 0., 0.), 0.25)
    self.assertEqual(bvnu(1., 0., 0.), 0.5)
    self.assertAlmostEqual(bvnu(0.43, 2.5, -1.0), 0.0062, places=4)
    self.assertAlmostEqual(bvnu(-0.17, 0.5, 1.0), 0.0351, places=4)
    self.assertAlmostEqual(bvnu(0., 0.5, 1.0), 0.0490, places=4)
    self.assertAlmostEqual(bvnu(-1., 0., -3.), 0.4987, places=4)
    self.assertAlmostEqual(bvnu(0., -5., -5.), 1., places=4)
    self.assertAlmostEqual(bvnu(0., 5., 5.), 0., places=4)
    self.assertAlmostEqual(bvnu(1., 3., 3.), 0.0013, places=4)
    self.assertAlmostEqual(bvnu(-1., 0., 0.), 0., places=4)

class TestBoundedIntegral(unittest.TestCase):
  
  def runTest(self):
    self.assertAlmostEqual(bvn(0.25, 0., 2.5, -1.2, 0.1), 0.1901, places=4)
    self.assertAlmostEqual(bvn(0., 0., 1., 0., 1.), 0.1165, places=4)
    self.assertAlmostEqual(bvn(0.5, 0., 1., 0., 1.), 0.1411, places=4)
    self.assertAlmostEqual(bvn(0.5, 0., np.inf, 0., 1.), 0.2059, places=4)


if __name__ == "__main__":
  unittest.main()