import time
from ASVD import *
from sklearn.model_selection import KFold
from computeError import *
from helper import *


def crossValidateASVD(ratings, k, c, n_users, n_movies):
    np.random.shuffle(ratings)
    kf = KFold(n_splits=5)
    ASVD_totalRMSE = 0.
    ASVD_totalMAE = 0.
    ASVD_totalTime = 0.

    for train_indices, test_indices in kf.split(ratings):
        # rozdziel oceny do testowania i trenowania
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stwórz macierz treningową
        mat_train = ratings2matrix(ratings_train, n_users, n_movies)

        # przybliżenie rzędu k [B1, B2] bazujące na Incremental ApproSVD
        mat_b1 = mat_train[:, :n1]
        mat_b2 = mat_train[:, n1:n1 + n2]

        # prawdopodobieństwo próbkowania kolumn dla B1
        nnz_b = np.count_nonzero(mat_b)
        p = np.zeros(n)
        for i in range(n):
            p[i] = np.count_nonzero(mat_b[:, i]) / float(nnz_b)

        t0 = time.clock()
        mat_hk = ASVD(mat_b1, c, k, p)
        t1 = time.clock() - t0
        mat_orig = mat_train[:, :n1 + n2]
        mat_ApproSVD = np.dot(np.dot(mat_hk, mat_hk.T), mat_orig)

        # przybliżenie rzędu k dla [B1, B2]
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        rating_pairs = createRatingPairs(ratings_test, mat_SVD, mat_ApproSVD, n1, n2)
        # policz RMSE
        ASVD_totalRMSE += computeRMSE(rating_pairs)
        # policz MAE
        ASVD_totalMAE += computeMAE(rating_pairs)

        # zmierz czas ASVD
        ASVD_totalTime += t1

    print('Total RMSE:', ASVD_totalRMSE)
    print('Total MAE:', ASVD_totalMAE)
    print('Total time:', ASVD_totalTime)
