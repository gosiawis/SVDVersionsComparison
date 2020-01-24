from incrementalApproSVD import *
import numpy as np
import unittest


class TestIncrementalSVD(unittest.TestCase):

    def testSVD(self):
        k = 5
        n1 = 10
        n2 = 5

        matrixB1 = np.random.randn(k, n1)
        matrixB2 = np.random.randn(k, n2)

        nonZeroB1 = np.count_nonzero(matrixB1)
        p1 = np.zeros(n1)
        for i in range(n1):
            p1[i] = np.count_nonzero(matrixB1[:, i]) / float(nonZeroB1)

        nonZeroB2 = np.count_nonzero(matrixB2)
        p2 = np.zeros(n2)
        for i in range(n2):
            p2[i] = np.count_nonzero(matrixB2[:, i]) / float(nonZeroB2)

        matrixHK = incrementalApproSVD(matrixB1, matrixB2, n1 - 5, n2 - 1, k, p1, p2)
        matrixA = np.hstack((matrixB1, matrixB2))
        np.testing.assert_array_almost_equal(np.dot(np.dot(matrixHK, matrixHK.T), matrixA), matrixA)


if __name__ == '__main__':
    unittest.main()
