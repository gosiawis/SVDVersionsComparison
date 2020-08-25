import numpy as np
import numpy.linalg as ln


def SVD(matrixM):
    """Apply SVD for a matrix

      :param matrixM: original matrix (m x n)
      """

    # get the number of rows and columns
    m = matrixM.shape[0]
    n = matrixM.shape[1]