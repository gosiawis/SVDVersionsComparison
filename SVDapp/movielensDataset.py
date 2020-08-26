import numpy as np

from checkAlgorithm import checkIASVD, checkISVD


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
    c1_IASVD, k_IASVD = checkIASVD(ratings, n_users, n_movies)
    n1_ISVD, k_ISVD = checkISVD(ratings, n_users, n_movies)
    print("MOVIELENS END")

    outMovielens = open("outMovielens.txt", "w")
    for test in [c1_IASVD, k_IASVD, n1_ISVD, k_ISVD]:
        for line in test:
            if type(line) is str:
                outMovielens.write(line)
                outMovielens.write("\n")
            else:
                outMovielens.write("RMSE: ")
                outMovielens.write(str(line[0]))
                outMovielens.write("\n")
                outMovielens.write("MAE: ")
                outMovielens.write(str(line[1]))
                outMovielens.write("\n")
                outMovielens.write("TIME: ")
                outMovielens.write(str(line[2]))
                outMovielens.write("\n")
    outMovielens.close()
