import numpy as np
import time
from ISVD import *
from sklearn.model_selection import KFold
from computeError import *


def ratings2matrix(ratings, n_users, n_movies):
    print("start function")
    data = np.zeros((n_users, n_movies))
    for rating in ratings:
        if len(rating) == 0:
            print("error")
            break
        # data[user_idx, movie_idx] = rating
        user_idx = int(rating[0]) - 1
        movie_idx = int(rating[1]) - 1
        if float(rating[2]) != 0.0:
            data[user_idx][movie_idx] = float(rating[2])
    print("For finish")
    if np.count_nonzero(data) != 0:
        return data
    return data


def createRatingPairs(ratings_test, mat1, mat2, n1, n2):
    rating_pairs = []
    ratings_test = map(list, ratings_test)
    for r in ratings_test:
        r = list(r)
        if len(r) == 0:
            continue
        if r[1] >= (n1 + n2):
            continue
        r1 = mat1[r[0] - 1, r[1] - 1]
        r2 = mat2[r[0] - 1, r[1] - 1]
        rating_pairs.append((r1, r2))
    return np.asarray(rating_pairs)


def crossValidateISVD(ratings, k, n1, n2, n_users, n_movies):
    np.random.shuffle(ratings)
    # kf = cross_validation.KFold(ratings.shape[0], n_folds=5)
    kf = KFold(n_splits=5)
    ISVD_totalRMSE = 0.
    ISVD_totalMAE = 0.
    ISVD_totalTime = 0.

    for train_indices, test_indices in kf.split(ratings):
        # rozdziel oceny do testowania i trenowania
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stwórz macierz treningową
        print(np.count_nonzero(list(ratings_train)))
        mat_train = ratings2matrix(map(list, ratings_train), n_users, n_movies)
        print(np.count_nonzero(mat_train))

        # rank-k approximation of [A1, A2] based on Incremental SVD
        mat_a1 = mat_train[:, :n1]
        mat_a2 = mat_train[:, n1:n1 + n2]

        t0 = time.clock()
        mat_u, mat_s, mat_vt = ISVD(mat_a1, mat_a2, k)
        t1 = time.clock() - t0
        mat_iSVD = np.dot(np.dot(mat_u, mat_s), mat_vt)

        # best rank-k approximation of [A1, A2] based on truncated SVD
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        rating_pairs = createRatingPairs(map(list, ratings_test), mat_SVD, mat_iSVD, n1, n2)
        # compute RMSE between SVD and Incremental SVD
        ISVD_totalRMSE += computeRMSE(rating_pairs)
        print(computeRMSE(rating_pairs))
        # compute MAE between SVD and Incremental SVD
        ISVD_totalMAE += computeMAE(rating_pairs)
        print(computeMAE(rating_pairs))
        # compute time elapsed on Incremental SVD
        ISVD_totalTime += t1

    print('Total RMSE:', ISVD_totalRMSE / 5.)
    print('Total MAE:', ISVD_totalMAE / 5.)
    print('Total time:', ISVD_totalTime / 5.)