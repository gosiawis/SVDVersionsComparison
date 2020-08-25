from crossValidateIASVD import *
from crossValidateISVD import *
import numpy as np
import matplotlib.pyplot as plt


def checkIASVD(ratings, n_users, n_movies):
    for c1 in [500, 600, 700, 800]:
        print("###################################################")
        print("Sprawdzenie krzyżowe IASVD dla c1=" + str(c1))
        crossValidateIASVD(ratings, k=10, n1=900, n2=100, c1=c1, c2=50, n_users=n_users, n_movies=n_movies)
        print("###################################################")
    for k in [10, 100, 400, 600]:
        print("###################################################")
        print("Sprawdzenie krzyżowe IASVD dla k=" + str(k))
        crossValidateIASVD(ratings, k=k, n1=900, n2=100, c1=800, c2=50, n_users=n_users, n_movies=n_movies)
        print("###################################################")


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

    print("MOVIELENS")
    checkIASVD(ratings, n_users, n_movies)

if __name__ == '__main__':
    movielens()
