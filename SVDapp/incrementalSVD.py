import numpy as np
import numpy.linalg as ln


def incrementalSVD(matrixA1, matrixA2, k, only_uk=False):
    """Apply SVD for a matrix with new columns

  :param only_uk:
  :param matrixA1: original matrix (m x n1)
  :param matrixA2: new columns (m x n2)
  :param k: rank-k for the approximated result
  :returns: rank-k approximated U, S, V^T as a result of svd([mat_a1, mat_a2])
  """

    if matrixA1.shape[0] != matrixA2.shape[0]:
        raise ValueError('Error: the number of rows both in mat_a1 and mat_a2 should be the same')

    # get the number of rows and columns
    m = matrixA1.shape[0]
    n1 = matrixA1.shape[1]
    n2 = matrixA2.shape[1]

    if k < 1:
        raise ValueError('Error: rank k must be greater than or equal to 1')
    if k > min(m, n1 + n2):
        raise ValueError('Error: rank k must be less than or equal to min(m, n1 + n2)')
    if not isinstance(k, int):
        raise ValueError('Error: rank k must be an integer')

    # apply SVD for the original matrix STEP 1
    matrixU1, singularVector1, matrixV1 = ln.svd(matrixA1, full_matrices=False)
    matrixS1 = np.diag(singularVector1)

    # define matrix F as [S, U^T A_1], and decompose it by SVD STEP 2
    matrixF = np.hstack((matrixS1, np.dot(matrixU1.T, matrixA2)))
    matrixUF, singularVectorF, matrixVFT = ln.svd(matrixF, full_matrices=False)

    # keep rank-k approximation
    matrixUF = matrixUF[:, :k]
    if only_uk:
        return np.dot(matrixU1, matrixUF)
    singularVectorF = singularVectorF[:k]  # keep values up to k value
    matrixVFT = matrixVFT[:k, :]

    # create a temporary matrix to compute matrix V_k
    V = matrixV1.T
    Z1 = np.zeros((n1, n2))  # if
    Z2 = np.zeros((n2, V.shape[1]))
    I = np.eye(n2)
    tempMatrix = np.vstack((
        np.hstack((V, Z1)),
        np.hstack((Z2, I))
    ))
    matrixVK = np.dot(tempMatrix, matrixVFT.T)

    # compute U_k and S_k
    matrixUK = np.dot(matrixU1, matrixUF)
    matrixSK = np.diag(singularVectorF)

    return matrixUK, matrixSK, matrixVK.T
