from crossValidateASVD import crossValidateASVD
from crossValidateIASVD import crossValidateIASVD
from crossValidateISVD import crossValidateISVD


def runIASVD(ratings, n_users, n_movies):
    c1_out = ["Sprawdzenie krzyzowe IASVD dla zmiennego c1"]
    for c1 in [500, 600, 700, 800]:
        pom = crossValidateIASVD(ratings, k=10, n1=900, n2=100, c1=c1, c2=50, n_users=n_users, n_movies=n_movies)
        c1_out.append(
            "900&100&\\textbf{" + str(c1) + "}&50&10&" + str(pom[0])[0:6] + "&" + str(pom[1])[0:6] + "&" + str(pom[2])[
                                                                                                           0:6] + "\\\\ \\hline")
    k_out = ["Sprawdzenie krzyzowe IASVD dla zmiennego k"]
    for k in [10, 100, 400, 600]:
        pom = crossValidateIASVD(ratings, k=k, n1=900, n2=100, c1=800, c2=50, n_users=n_users, n_movies=n_movies)
        k_out.append(
            "900&100&800&50&\\textbf{" + str(k) + "}&" + str(pom[0])[0:6] + "&" + str(pom[1])[0:6] + "&" + str(pom[2])[
                                                                                                           0:6] + "\\\\ \\hline")
    return c1_out, k_out


def runISVD(ratings, n_users, n_movies):
    n1_out = ["Sprawdzenie krzyzowe ISVD dla zmiennego n1"]
    for n1 in [500, 600, 700, 800]:
        pom = crossValidateISVD(ratings, k=10, n1=n1, n2=50, n_users=n_users, n_movies=n_movies)
        n1_out.append(
            "\\textbf{" + str(n1) + "}&50&10&" + str(pom[0])[0:6] + "&" + str(pom[1])[0:6] + "&" + str(pom[2])[
                                                                                                   0:6] + "\\\\ \\hline")
    k_out = ["Sprawdzenie krzyzowe ISVD dla zmiennego k"]
    for k in [10, 100, 400, 600]:
        pom = crossValidateISVD(ratings, k=k, n1=800, n2=50, n_users=n_users, n_movies=n_movies)
        k_out.append("800&50&\\textbf{" + str(k) + "}&" + str(pom[0])[0:6] + "&" + str(pom[1])[0:6] + "&" + str(pom[2])[
                                                                                                            0:6] + "\\\\ \\hline")
    return n1_out, k_out


def runASVD(ratings, n_users, n_movies):
    c_out = ["Sprawdzenie krzyzowe ISVD dla zmiennego n1"]
    for c in [500, 600, 700, 800]:
        pom = crossValidateASVD(ratings, k=10, c=c, n_users=n_users, n_movies=n_movies)
        c_out.append(
            "\\textbf{" + str(c) + "}&10&" + str(pom[0])[0:6] + "&" + str(pom[1])[0:6] + "&" + str(pom[2])[
                                                                                               0:6] + "\\\\ \\hline")
    k_out = ["Sprawdzenie krzyzowe ISVD dla zmiennego k"]
    for k in [10, 100, 400, 600]:
        pom = crossValidateASVD(ratings, k=k, c=800, n_users=n_users, n_movies=n_movies)
        k_out.append("800&\\textbf{" + str(k) + "}&" + str(pom[0])[0:6] + "&" + str(pom[1])[0:6] + "&" + str(pom[2])[
                                                                                                         0:6] + "\\\\ \\hline")
    return c_out, k_out


def runAll(ratings, n_users, n_movies, dataset_name, filename):
    print(dataset_name, " START")
    c1_IASVD, k_IASVD = runIASVD(ratings, n_users, n_movies)
    n1_ISVD, k_ISVD = runISVD(ratings, n_users, n_movies)
    c_ASVD, k_ASVD = runASVD(ratings, n_users, n_movies)
    print(dataset_name, " END")

    out = open(filename, "w")
    for test in [c1_IASVD, k_IASVD, n1_ISVD, k_ISVD, c_ASVD, k_ASVD]:
        for line in test:
            if type(line) is str:
                out.write(line)
                out.write("\n")
    out.close()
