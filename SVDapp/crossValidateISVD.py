import time
from ISVD import *
from sklearn.model_selection import KFold
from computeError import *
from helper import *


def crossValidateISVD(ratings, k, n1, n2, n_users, n_movies):
    np.random.shuffle(ratings)
    kf = KFold(n_splits=5)
    ISVD_totalRMSE = 0.
    ISVD_totalMAE = 0.

    t0 = time.clock()
    for train_indices, test_indices in kf.split(ratings):
        # rozdziel oceny do testowania i trenowania
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stworz macierz treningowa
        mat_train = ratings2matrix(ratings_train, n_users, n_movies)

        # przyblizenie rzedu k dla [A1, A2] bazujace na ISVD
        mat_a1 = mat_train[:, :n1]
        mat_a2 = mat_train[:, n1:n1 + n2]

        mat_u, mat_s, mat_vt = ISVD(mat_a1, mat_a2, k)
        mat_iSVD = np.dot(np.dot(mat_u, mat_s), mat_vt)

        # przyblizenie rzedu k dla [A1, A2]
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        rating_pairs = createRatingPairs(ratings_test, mat_SVD, mat_iSVD, n1, n2)
        # policz RMSE
        ISVD_totalRMSE += computeRMSE(rating_pairs)
        # policz MAE
        ISVD_totalMAE += computeMAE(rating_pairs)

    # zmierz czas ISVD
    ISVD_totalTime = time.clock() - t0

    return [ISVD_totalRMSE, ISVD_totalMAE, ISVD_totalTime]
