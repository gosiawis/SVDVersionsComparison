import time
from algorithmIASVD import *
from sklearn.model_selection import KFold
from computeError import *
from helper import *


def crossValidateIASVD(ratings, k, n1, n2, c1, c2, n_users, n_movies):
    np.random.shuffle(ratings)
    kf = KFold(n_splits=5)
    IASVD_totalRMSE = 0.
    IASVD_totalMAE = 0.

    t0 = time.clock()
    for train_indices, test_indices in kf.split(ratings):
        # rozdziel oceny do testowania i trenowania
        print("TEST: ", test_indices, "TRAIN: ", train_indices)
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stworz macierz treningowa
        mat_train = ratings2matrix(ratings_train, n_users, n_movies)

        # przyblizenie rzedu k [B1, B2] bazujace na Incremental ApproSVD
        mat_b1 = mat_train[:, :n1]
        mat_b2 = mat_train[:, n1:n1 + n2]

        # prawdopodobienstwo probkowania kolumn dla B1
        nnz_b1 = np.count_nonzero(mat_b1)
        p1 = np.zeros(n1)
        for i in range(n1):
            p1[i] = np.count_nonzero(mat_b1[:, i]) / float(nnz_b1)

        # prawdopodobienstwo probkowania kolumn dla B2
        nnz_b2 = np.count_nonzero(mat_b2)
        p2 = np.zeros(n2)
        for i in range(n2):
            p2[i] = np.count_nonzero(mat_b2[:, i]) / float(nnz_b2)

        mat_hk = IASVD(mat_b1, mat_b2, c1, c2, k, p1, p2)

        mat_orig = mat_train[:, :n1 + n2]
        mat_ApproSVD = np.dot(np.dot(mat_hk, mat_hk.T), mat_orig)

        # przyblizenie rzędu k dla [B1, B2]
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        rating_pairs = createRatingPairs(ratings_test, mat_SVD, mat_ApproSVD, n1, n2)
        # policz RMSE
        IASVD_totalRMSE += computeRMSE(rating_pairs)
        # policz MAE
        IASVD_totalMAE += computeMAE(rating_pairs)

    # zmierz czas IASVD
    IASVD_totalTime = time.clock() - t0

    return [IASVD_totalRMSE, IASVD_totalMAE, IASVD_totalTime]
