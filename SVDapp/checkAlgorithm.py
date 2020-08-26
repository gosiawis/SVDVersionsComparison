from crossValidateIASVD import crossValidateIASVD
from crossValidateISVD import crossValidateISVD


def checkIASVD(ratings, n_users, n_movies):
    c1_out = []
    for c1 in [500, 600, 700, 800]:
        c1_out.append("Sprawdzenie krzyzowe IASVD dla c1=" + str(c1))
        c1_out.append(
            crossValidateIASVD(ratings, k=10, n1=900, n2=100, c1=c1, c2=50, n_users=n_users, n_movies=n_movies))
    k_out = []
    for k in [10, 100, 400, 600]:
        k_out.append("Sprawdzenie krzyzowe IASVD dla k=" + str(k))
        k_out.append(
            crossValidateIASVD(ratings, k=k, n1=900, n2=100, c1=800, c2=50, n_users=n_users, n_movies=n_movies))
    return c1_out, k_out


def checkISVD(ratings, n_users, n_movies):
    n1_out = []
    for n1 in [500, 600, 700, 800]:
        n1_out.append("Sprawdzenie krzyzowe ISVD dla n1=" + str(n1))
        n1_out.append(crossValidateISVD(ratings, k=10, n1=n1, n2=50, n_users=n_users, n_movies=n_movies))
    k_out = []
    for k in [10, 100, 400, 600]:
        k_out.append("Sprawdzenie krzyzowe ISVD dla k=" + str(k))
        k_out.append(crossValidateISVD(ratings, k=k, n1=800, n2=50, n_users=n_users, n_movies=n_movies))
    return n1_out, k_out
