import numpy as np

from runAlgorithm import runAll


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

    runAll(ratings, n_users, n_movies, dataset_name="MOVIELENS", filename="outMovielens.txt")
