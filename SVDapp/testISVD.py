from ISVD import *
import numpy as np
import unittest

class TestISVD(unittest.TestCase):

  def testSVD(self):
    matrixA1 = np.random.randn(2, 3)
    matrixA2 = np.random.randn(2, 1)
    k = 2 # same as the original
    matrixU, matrixS, matrixVT = ISVD(matrixA1, matrixA2, k)
    np.testing.assert_array_almost_equal(np.dot(np.dot(matrixU, matrixS), matrixVT), np.hstack((matrixA1, matrixA2)))

  def testInvalidK(self):
    self.assertRaises(ValueError, ISVD, np.random.randn(2, 3), np.random.randn(2, 1), 0)
    self.assertRaises(ValueError, ISVD, np.random.randn(2, 3), np.random.randn(2, 1), 100)

  def testDifferentM(self):
    self.assertRaises(ValueError, ISVD, np.random.randn(2, 3), np.random.randn(3, 1), 2)

if __name__ == '__main__':
  unittest.main()
  TestISVD.testSVD()