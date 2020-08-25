import numpy as np
import numpy.linalg as ln


def checkMatrixShapes(matrixA1, matrixA2):
    if matrixA1.shape[0] != matrixA2.shape[0]:
        raise ValueError('Błąd: liczba wierszy w macierzy A1 i A2 powinna być taka sama')


def checkParams(m, n1, n2, k):
    if k < 1:
        raise ValueError('Błąd: rząd k musi być równy lub większy od 1')
    if k > min(m, n1 + n2):
        raise ValueError('Błąd: rząd k musi być równy lub mniejszy od min(m, n1 + n2)')
    if not isinstance(k, int):
        raise ValueError('Błąd: rząd k musi być liczbą całkowitą')


def ISVD(matrixA1, matrixA2, k, only_uk=False):
    """Zastosuj Incremental SVD dla macierzy z nowymi kolumnami

  :param only_uk: przy wartości True funkcja zwraca tylko macież U
  :param matrixA1: oryginalna macierz (m x n1)
  :param matrixA2: nowe kolumny (m x n2)
  :param k: rząd przybliżenia
  :returns: przybliżenie rzędu k U, S, V^T jako wynik svd([mat_a1, mat_a2])
  """
    # sprawdź czy liczba wierszy A1 i A2 jest taka sama
    checkMatrixShapes(matrixA1, matrixA2)

    # znajdź rozmiar macierzy
    m = matrixA1.shape[0]  # m - liczba wierszy macierzy B1 i B2
    n1 = matrixA1.shape[1]  # n1 - liczba kolumn macierzy B1
    n2 = matrixA2.shape[1]  # n2 - liczba kolumn macierzy B2

    # sprawdź czy parametry spełniają założenia 1<=k<=min(m, n1+n2)
    checkParams(m, n1, n2, k)

    # zastosuj SVD na oryginalnej macierzy A1
    matrixU1, singularVector1, matrixV1 = ln.svd(matrixA1, full_matrices=False)
    matrixS1 = np.diag(singularVector1)

    # zdefiniuj macierz F jako [S, U^T A_1] i zastosuj na niej SVD
    matrixF = np.hstack((matrixS1, np.dot(matrixU1.T, matrixA2)))
    matrixUF, singularVectorF, matrixVFT = ln.svd(matrixF, full_matrices=False)

    # zachowaj przybliżenie do rzędu k
    matrixUF = matrixUF[:, :k]
    if only_uk:
        return np.dot(matrixU1, matrixUF)
    singularVectorF = singularVectorF[:k]  # zachowaj wartości do k
    matrixVFT = matrixVFT[:k, :]

    # stwórz tymczasową macierz do obliczenia Vk
    V = matrixV1.T
    Z1 = np.zeros((n1, n2))
    Z2 = np.zeros((n2, V.shape[1]))
    I = np.eye(n2)
    tempMatrix = np.vstack((
        np.hstack((V, Z1)),
        np.hstack((Z2, I))
    ))
    matrixVK = np.dot(tempMatrix, matrixVFT.T)

    # obilcz macierz Uk i macierz Sk
    matrixUK = np.dot(matrixU1, matrixUF)
    matrixSK = np.diag(singularVectorF)

    return matrixUK, matrixSK, matrixVK.T
