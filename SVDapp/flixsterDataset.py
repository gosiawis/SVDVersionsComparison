import numpy as np
from collections import Counter

from checkAlgorithm import checkIASVD, checkISVD


def sampleDataset(ratings, u, m):
    # Zasady ograniczenia zbioru:
    # - uzytkownik ocenil wiecej niz 250 filmow
    # - film ma wiecej niz 30 ocen
    list_movies = []
    list_user = []
    for r in ratings:
        list_movies.append(r[1])
        list_user.append(r[0])
    movie_counter = Counter(list_movies)
    user_counter = Counter(list_user)
    print(ratings.shape[0])
    ratings_wout_movies = []
    for r in ratings:
        if movie_counter[r[1]] < 30:
            ratings_wout_movies.append(r)
    ratings_wout_users = []
    for r in ratings_wout_movies:
        if user_counter[r[0]] < 250:
            ratings_wout_users.append(r)
    print(len(ratings_wout_users))
    return ratings_wout_users


def flixster():
    # zbior danych Flixster pobrany ze strony: https://sites.google.com/view/mohsenjamali/home
    PATH = 'flixster/Ratings.timed.txt'
    # dane zawieraja oceny od 786936 uzytkownikow dla 48794 filmow
    n_ratings = 8196077
    n_users = 786936
    n_movies = 48794

    ratings = []
    with open(PATH) as f:
        for i in range(n_ratings):
            line = f.readline()
            if i == 0:
                continue
            else:
                element = list(line.rstrip().split('\t'))
                if len(element) > 1:
                    ratings.append([float(element[0]), int(element[1]), float(element[2])])
    ratings = np.asarray(ratings)

    ratings = sampleDataset(ratings, n_users, n_movies)
    n_users = 8465
    n_movies = 9602

    print("FLIXSTER START")
    c1_IASVD, k_IASVD = checkIASVD(ratings, n_users, n_movies)
    n1_ISVD, k_ISVD = checkISVD(ratings, n_users, n_movies)
    print("FLIXSTER END")

    outFlixster = open("outFlixster.txt", "w")
    for test in [c1_IASVD, k_IASVD, n1_ISVD, k_ISVD]:
        for line in test:
            if type(line) is str:
                outFlixster.write(line)
                outFlixster.write("\n")
            else:
                outFlixster.write("RMSE: ")
                outFlixster.write(str(line[0]))
                outFlixster.write("\n")
                outFlixster.write("MAE: ")
                outFlixster.write(str(line[1]))
                outFlixster.write("\n")
                outFlixster.write("TIME: ")
                outFlixster.write(str(line[2]))
                outFlixster.write("\n")
    outFlixster.close()
