import numpy as np
import numpy.linalg as ln


def ASVD(matrixM, c, k, p):
    """Zastosuj ApproSVD dla macierzy M

  :param matrixM: oryginalna macierz (m x n)
  :param c: liczba próbkowanych kolumn z macierzy A
  :param k: rząd przybliżenia
  :param p: prawdopodobieństwo próbkowania dla każdej kolumny macierzy M
  :returns: H_k macierz jako wartość wyjściowa ApproSVD (H_k H_k^T = I)
  """

    if len(p[p < 0]) != 0:
        raise ValueError('Błąd: ujemne prawdopodobieństwo p jest poza zakresem')

    if not np.isclose(sum(p), 1.):
        raise ValueError('Błąd: suma prawdopodobieństw musi być równa 1 dla p')

    # znajdź rozmiar macierzy
    m = matrixM.shape[0]  # m - liczba wierszy macierzy M
    n = matrixM.shape[1]  # n - liczba kolumn macierzy M

    if c < m:
        raise ValueError('Błąd: c musi być równe lub mniejsze od m')

    if k < 1:
        raise ValueError('Błąd: rząd k musi być równy lub większy od 1')
    if k < c:
        raise ValueError('Błąd: rzad k musi być większy lub równy c')
    if k < m:
        raise ValueError('Błąd: rząd k musi być większy lub równy m')

    # próbkuj c kolumn z macierzy A i utwórz z nich macierz C
    matrixC = np.zeros((m, c))
    samples = np.random.choice(range(n), c, replace=False, p=p)
    for t in range(c):
        matrixC[:, t] = matrixM[:, samples[t]] / np.sqrt(c * p[samples[t]])

    # zastosuj SVD na oryginalnej macierzy
    matrixU, singularVector, matrixV = ln.svd(matrixC, full_matrices=False)
    matrixS = np.diag(singularVector)

    # keep rank-k approximation
    matrixUK = matrixU[:, :k]
    singularVectorK = singularVector[:k]  # keep values up to k value
    matrixVK = matrixV[:k, :]

    matrixSK = np.diag(singularVectorK)

    return matrixUK, matrixSK, matrixVK.T