from ISVD import *
import numpy as np


def checkMatrixShape(matrixB1, matrixB2):
    if matrixB1.shape[0] != matrixB2.shape[0]:
        raise ValueError('Błąd: liczba wierszy w macierzy B1 i B2 powinna być taka sama')


def checkProbabilities(p1, p2):
    if len(p1[p1 < 0]) != 0:
        raise ValueError('Błąd: ujemne prawdopodobieństwo p1 jest poza zakresem')
    if len(p2[p2 < 0]) != 0:
        raise ValueError('Błąd: ujemne prawdopodobieństwo p2 jest poza zakresem')

    if not np.isclose(sum(p1), 1.):
        raise ValueError('Błąd: suma prawdopodobieństw musi być równa 1 dla p1')
    if not np.isclose(sum(p2), 1.):
        raise ValueError('Błąd: suma prawdopodobieństw musi być równa 1 dla p2')


def checkParameters(m, n1, n2, k, c1, c2):
    if c1 >= n1:
        raise ValueError('Błąd: c1 musi być mniejsze od n1')
    if c2 >= n2:
        raise ValueError('Błąd: c2 musi być mniejsze od n2')

    if k < 1:
        raise ValueError('Błąd: rząd k musi być równy lub większy od 1')
    if k > min(m, c1 + c2):
        raise ValueError('Błąd: rząd k musi być równy lub mniejszy od min(m, n1 + n2)')
    if not isinstance(k, int):
        raise ValueError('Błąd: rząd k musi być liczbą całkowitą')


def IASVD(matrixB1, matrixB2, c1, c2, k, p1, p2):
    """Zastosuj Incremental ApproSVD na macierzy z nowymi kolumnami

  :param matrixB1: oryginalna macierz (m x n1)
  :param matrixB2: nowe kolumny (m x n2)
  :param c1: liczba próbkowanych kolumn z macierzy B1
  :param c2: liczba próbkowanych kolumn z macierzy B2
  :param k: rząd przybliżenia
  :param p1: prawdopodobieństwo próbkowania dla każdej kolumny macierzy B1
  :param p2: prawdopodobieństwo próbkowania dla każdej kolumny macierzy B2
  :returns: H_k macierz jako wartość wyjściowa Incremental ApproSVD (H_k H_k^T = I)
  """
    # sprawdź czy liczba wierszy B1 i B2 jest taka sama
    checkMatrixShape(matrixB1, matrixB2)
    # sprawdź czy wartość prawdopodobieństw jest zgodna z założeniami
    checkProbabilities(p1, p2)

    # znajdź rozmiar macierzy
    m = matrixB1.shape[0]  # m - liczba wierszy macierzy B1 i B2
    n1 = matrixB1.shape[1]  # n1 - liczba kolumn macierzy B1
    n2 = matrixB2.shape[1]  # n2 - liczba kolumn macierzy B2

    # sprawdź czy wartość parametrów jest zgodna z założeniami: c1>=n1, c2>=n2, 1<=k<=min(m, c1+c2)
    checkParameters(m, n1, n2, k, c1, c2)

    # próbkuj c1 kolumn z macierzy B1 i utwórz z nich macierz C1
    matrixC1 = np.zeros((m, c1))
    samples = np.random.choice(range(n1), c1, replace=False, p=p1)
    for t in range(c1):
        matrixC1[:, t] = matrixB1[:, samples[t]] / np.sqrt(c1 * p1[samples[t]])

    # próbkuj c2 kolumn z macierzy B2 i utwórz z nich macierz C2
    matrixC2 = np.zeros((m, c2))
    samples = np.random.choice(range(n2), c2, replace=False, p=p2)
    for t in range(c2):
        matrixC2[:, t] = matrixB2[:, samples[t]] / np.sqrt(c2 * p2[samples[t]])

    # zastosuj Incremental SVD dla mniejszych macierzy C1, C2 i weź tylko U_k jako H_k
    return ISVD(matrixC1, matrixC2, k, True)
