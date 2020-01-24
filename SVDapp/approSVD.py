from incrementalSVD import *
import numpy as np
import numpy.linalg as ln
import sys


def approSVD(matrixA, c, k, p):
    """Apply ApproSVD for a matrix with new columns

  :param matrixA: original matrix (m x n)
  :param c: the number of sampled columns from A
  :param k: rank-k for the approximated result
  :param p: sampling probabilities for each column in A
  :returns: H_k as an output of ApproSVD (H_k H_k^T = I)
  """

    if len(p[p < 0]) != 0:
        raise ValueError('Error: negative probabilities in p are not allowed')

    if not np.isclose(sum(p), 1.):
        raise ValueError('Error: sum of the probabilities must be 1 for p')

    # get the number of rows and columns
    m = matrixA.shape[0]
    n = matrixA.shape[1]

    if c <= m:
        raise ValueError('Error: c must be less than m')

    if k < 1:
        raise ValueError('Error: rank k must be greater than or equal to 1')
    if k < c:
        raise ValueError('Error: rank k must be greater than or equal to c')
    if k < m:
        raise ValueError('Error: rank k must be greater than or equal to m')


    # sample c1 columns from A, and combine them as a matrix C
    matrixC = np.zeros((m, c))
    samples = np.random.choice(range(n), c, replace=False, p=p)
    for t in range(c):
        matrixC[:, t] = matrixA[:, samples[t]] / np.sqrt(c * p[samples[t]])

    # apply SVD for the original matrix STEP 1
    matrixU, singularVector, matrixV = ln.svd(matrixC, full_matrices=False)
    matrixS = np.diag(singularVector)

    # keep rank-k approximation
    matrixUK = matrixU[:, :k]
    singularVectorK = singularVector[:k]  # keep values up to k value
    matrixVK = matrixV[:k, :]

    matrixSK = np.diag(singularVectorK)

    return matrixUK, matrixSK, matrixVK.T
