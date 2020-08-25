import numpy as np
import time
from IASVD import *
from sklearn.model_selection import KFold
from computeError import *


def ratings2matrix(ratings, n_users, n_movies):
    print("start function")
    # print(list(ratings))
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


def crossValidateIASVD(ratings, k, n1, n2, c1, c2, n_users, n_movies):
    np.random.shuffle(ratings)
    # kf = cross_validation.KFold(ratings.shape[0], n_folds=5)
    kf = KFold(n_splits=5)
    IASVD_totalRMSE = 0.
    IASVD_totalMAE = 0.
    IASVD_totalTime = 0.

    for train_indices, test_indices in kf.split(ratings):
        print("test kf")
        print("TRAIN:", train_indices, "TEST:", test_indices)
        # rozdziel oceny do testowania i trenowania
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stwórz macierz treningową
        print(np.count_nonzero(list(ratings_train)))
        pom = map(list, ratings_train)
        print(pom)
        mat_train = ratings2matrix(map(list, ratings_train), n_users, n_movies)
        print(np.count_nonzero(mat_train))

        '''# przybliżenie rzędu k [B1, B2] bazujące na Incremental ApproSVD
        mat_b1 = mat_train[:, :n1]
        mat_b2 = mat_train[:, n1:n1 + n2]

        # prawdopodobieństwo próbkowania kolumn dla B1
        nnz_b1 = np.count_nonzero(mat_b1)
        p1 = np.zeros(n1)
        for i in range(n1):
            p1[i] = np.count_nonzero(mat_b1[:, i]) / float(nnz_b1)

        # prawdopodobieństwo próbkowania kolumn dla B2
        nnz_b2 = np.count_nonzero(mat_b2)
        p2 = np.zeros(n2)
        for i in range(n2):
            p2[i] = np.count_nonzero(mat_b2[:, i]) / float(nnz_b2)

        t0 = time.clock()
        mat_hk = IASVD(mat_b1, mat_b2, c1, c2, k, p1, p2)
        t1 = time.clock() - t0
        mat_orig = mat_train[:, :n1 + n2]
        mat_ApproSVD = np.dot(np.dot(mat_hk, mat_hk.T), mat_orig)

        # best rank-k approximation of [B1, B2] based on truncated SVD
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        rating_pairs = createRatingPairs(map(list, ratings_test), mat_SVD, mat_ApproSVD, n1, n2)
        # compute RMSE between SVD and Incremental ApproSVD
        IASVD_totalRMSE += computeRMSE(rating_pairs)
        print(computeRMSE(rating_pairs))
        # compute MAE between SVD and Incremental ApproSVD
        IASVD_totalMAE += computeMAE(rating_pairs)
        print(computeMAE(rating_pairs))
        # compute time elapsed on Incremental ApproSVD
        IASVD_totalTime += t1

    print('Total RMSE:', IASVD_totalRMSE / 5.)
    print('Total MAE:', IASVD_totalMAE / 5.)
    print('Total time:', IASVD_totalTime / 5.)'''