from crossValidateIASVD import *
from crossValidateISVD import *
import numpy as np


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
            ratings.append(map(int, line.rstrip().split('\t')))
    ratings = np.asarray(ratings)

    print("MOVIELENS")
    for c1 in [500, 600, 700, 800]:
        print("Sprawdzenie krzyżowe IASVD dla c1=" + str(c1))
        crossValidateIASVD(ratings, k=10, n1=900, n2=100, c1=c1, c2=50, n_users=n_users, n_movies=n_movies)
        print("Sprawdzenie krzyżowe ISVD dla n1=" + str(c1))
        crossValidateISVD(ratings, k=10, n1=c1, n2=50, n_users=n_users, n_movies=n_movies)

    for k in [10, 100, 400, 900]:
        print("Sprawdzenie krzyżowe IASVD dla k=" + str(k))
        crossValidateIASVD(ratings, k=k, n1=900, n2=100, c1=800, c2=50, n_users=n_users, n_movies=n_movies)
        print("Sprawdzenie krzyżowe ISVD dla k=" + str(k))
        crossValidateISVD(ratings, k=k, n1=800, n2=50, n_users=n_users, n_movies=n_movies)

if __name__ == '__main__':
    movielens()
