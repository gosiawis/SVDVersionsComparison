import numpy as np
from collections import Counter

from runAlgorithm import runAll


def sampleDataset(ratings):
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
    ratings_wout_movies = []
    for r in ratings:
        if movie_counter[r[1]] < 30:
            ratings_wout_movies.append(r)
    ratings_wout_users = []
    for r in ratings_wout_movies:
        if user_counter[r[0]] < 250:
            ratings_wout_users.append(r)
    # popraw indeksowanie dla uzytkownikow
    index_map_user = []
    for r in ratings_wout_users:
        if r[0] not in index_map_user:
            index_map_user.append(r[0])
    for r in ratings_wout_users:
        r[0] = index_map_user.index(r[0])
    index_map_movie = []
    for r in ratings_wout_users:
        if r[1] not in index_map_movie:
            index_map_movie.append(r[1])
    for r in ratings_wout_users:
        r[1] = index_map_movie.index(r[1])
    print("Filmy: ", len(index_map_movie))
    print("Użytkownicy: ", len(index_map_user))
    return ratings_wout_users, len(index_map_movie), len(index_map_user)


def flixster():
    # zbior danych Flixster pobrany ze strony: https://sites.google.com/view/mohsenjamali/home
    PATH = 'flixster/Ratings.timed.txt'
    # dane zawieraja oceny od 786936 uzytkownikow dla 48794 filmow
    n_ratings = 8196077

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

    ratings, n_movies, n_users = sampleDataset(ratings)
    ratings = np.asarray(ratings)

    runAll(ratings, n_users, n_movies, dataset_name="FLIXSTER", filename="outFlixster.txt")
