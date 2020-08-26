import numpy as np


def ratings2matrix(ratings, n_users, n_movies):
    data = np.zeros((n_users, n_movies))
    for rating in ratings:
        rating = list(rating)
        if len(rating) == 0:
            continue
        # data[user_idx, movie_idx] = rating
        user_idx = int(rating[0]) - 1
        movie_idx = int(rating[1]) - 1
        if float(rating[2]) != 0.0:
            data[user_idx][movie_idx] = float(rating[2])
    return data


def createRatingPairs(ratings_test, mat1, mat2, n1, n2):
    rating_pairs = []
    ratings_test = map(list, ratings_test)
    for r in ratings_test:
        r = list(r)
        if len(r) == 0:
            continue
        if int(r[1]) >= (n1 + n2):
            continue
        r1 = mat1[int(r[0]) - 1, int(r[1]) - 1]
        r2 = mat2[int(r[0]) - 1, int(r[1]) - 1]
        rating_pairs.append((r1, r2))
    return np.asarray(rating_pairs)
