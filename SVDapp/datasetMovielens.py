import numpy as np

from runAlgorithm import runIASVD, runISVD


def movielens():
    # zbiór danych MovieLens 100k pobrany ze strony: http://grouplens.org/datasets/movielens/
    PATH = 'movielens/u.data'
    # dane zawierają oceny od 943 użytkowników dla 1682 filmów
    n_ratings = 100000
    n_users = 943
    n_movies = 1682

    ratings = []
    with open(PATH) as f:
        for i in range(n_ratings):
            line = f.readline()
            ratings.append(list(line.rstrip().split('\t')))
    ratings = np.asarray(ratings)

    print("MOVIELENS START")
    c1_IASVD, k_IASVD = runIASVD(ratings, n_users, n_movies)
    n1_ISVD, k_ISVD = runISVD(ratings, n_users, n_movies)
    print("MOVIELENS END")

    outMovielens = open("outMovielens.txt", "w")
    for test in [c1_IASVD, k_IASVD, n1_ISVD, k_ISVD]:
        for line in test:
            if type(line) is str:
                outMovielens.write(line)
                outMovielens.write("\n")
    outMovielens.close()
