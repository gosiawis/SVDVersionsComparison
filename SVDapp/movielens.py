from ISVD import *
from IASVD import *
from sklearn.model_selection import KFold
import numpy as np

# zbiór danych MovieLens 100k pobrany ze strony: http://grouplens.org/datasets/movielens/
PATH = 'movielens/u.data'
n_users = 943
n_movies = 1682


def computeMAE(rating_pairs):
    n = rating_pairs.shape[0]
    total = 0
    for rt, rp in rating_pairs:
        total += abs(rt - rp)
    return total / float(n)


def computeRMSE(rating_pairs):
    n = rating_pairs.shape[0]
    # print(n)
    total = 0
    for rt, rp in rating_pairs:
        total += (rt - rp) ** 2
    RMSE = np.sqrt(total / float(n))
    # print(RMSE)
    return RMSE


def ratings2matrix(ratings):
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
        if len(r) == 0:
            break
        if r[1] >= (n1 + n2):
            continue
        r1 = mat1[r[0] - 1, r[1] - 1]
        r2 = mat2[r[0] - 1, r[1] - 1]
        rating_pairs.append((r1, r2))
    return np.asarray(rating_pairs)

'''
def crossValidate(ratings):
    # zmniejsz rząd
    k = 10

    # liczba oryginalnych kolumn Incremental ApproSVD
    # B1: 900, B2: 100
    n1 = 900
    n2 = 100

    # ile kolumn zostało sprawdzonych? (n_movies=1682 jest najwyższą możliwą wartością)
    # A1: 500, A2: 50
    c1 = 500
    c2 = 50

    np.random.shuffle(ratings)
    # kf = cross_validation.KFold(ratings.shape[0], n_folds=5)
    kf = KFold(n_splits=5)
    print(kf)

    ISVD_totalRMSE = 0.
    IASVD_totalRMSE = 0.

    for train_indices, test_indices in kf.split(ratings):

        # rozdziel oceny do testowania i trenowania
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stwórz macierz treningową
        mat_train = ratings2matrix(ratings_train.tolist())

        # przybliżenie rzędu k [B1, B2] bazujące na Incremental ApproSVD
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

        mat_hk = IASVD(mat_b1, mat_b2, c1, c2, k, p1, p2)
        mat_orig = mat_train[:, :n1 + n2]
        mat_ApproSVD = np.dot(np.dot(mat_hk, mat_hk.T), mat_orig)

        # best rank-k approximation of [B1, B2] based on truncated SVD
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        # compute RMSE between SVD and ApproSVD
        rating_pairs = createRatingPairs(ratings_test, mat_SVD, mat_ApproSVD, n1, n2)
        IASVD_totalRMSE += computeRMSE(rating_pairs)

        # rank-k approximation of [A1, A2] based on Incremental SVD
        mat_a1 = mat_train[:, :c1]
        mat_a2 = mat_train[:, c1:c1 + c2]

        mat_u, mat_s, mat_vt = ISVD(mat_a1, mat_a2, k)
        mat_iSVD = np.dot(np.dot(mat_u, mat_s), mat_vt)

        # best rank-k approximation of [A1, A2] based on truncated SVD
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :c1 + c2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        # compute RMSE between SVD and iSVD
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        rating_pairs = createRatingPairs(ratings_test, mat_SVD, mat_iSVD, c1, c2)
        ISVD_totalRMSE += computeRMSE(rating_pairs)

    print('Incremental SVD      vs. truncated SVD:', ISVD_totalRMSE / 5.)
    print('Incremental ApproSVD vs. truncated SVD:', IASVD_totalRMSE / 5.)
'''

def crossValidateIASVD(ratings, k, n1, n2, c1, c2):
    np.random.shuffle(ratings)
    # kf = cross_validation.KFold(ratings.shape[0], n_folds=5)
    kf = KFold(n_splits=5)
    IASVD_totalRMSE = 0.

    for train_indices, test_indices in kf.split(ratings):
        print("test kf")
        print("TRAIN:", train_indices, "TEST:", test_indices)
        # rozdziel oceny do testowania i trenowania
        ratings_train = ratings[train_indices]
        ratings_test = ratings[test_indices]

        # stwórz macierz treningową
        print(np.count_nonzero(list(ratings_train)))
        mat_train = ratings2matrix(map(list, ratings_train))
        print(np.count_nonzero(mat_train))

        # przybliżenie rzędu k [B1, B2] bazujące na Incremental ApproSVD
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

        mat_hk = IASVD(mat_b1, mat_b2, c1, c2, k, p1, p2)
        mat_orig = mat_train[:, :n1 + n2]
        mat_ApproSVD = np.dot(np.dot(mat_hk, mat_hk.T), mat_orig)

        # best rank-k approximation of [B1, B2] based on truncated SVD
        mat_u, vec_s, mat_vt = ln.svd(mat_train[:, :n1 + n2], full_matrices=False)
        mat_SVD = np.dot(np.dot(mat_u[:, :k], np.diag(vec_s[:k])), mat_vt[:k, :])

        # compute RMSE between SVD and ApproSVD
        rating_pairs = createRatingPairs(ratings_test, mat_SVD, mat_ApproSVD, n1, n2)
        IASVD_totalRMSE += computeRMSE(rating_pairs)

    print('Incremental ApproSVD vs. truncated SVD:', IASVD_totalRMSE / 5.)


def main():
    # the data has ratings for 943 users on 1682 movies
    n_ratings = 100000

    ratings = []
    with open(PATH) as f:
        for i in range(n_ratings):
            line = f.readline()
            ratings.append(map(int, line.rstrip().split('\t')))
    ratings = np.asarray(ratings)

    crossValidateIASVD(ratings, k=10, n1=900, n2=100, c1=500, c2=50)


if __name__ == '__main__':
    main()
