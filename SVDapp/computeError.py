import numpy as np


def computeRMSE(rating_pairs):
    n = rating_pairs.shape[0]
    # print(n)
    total = 0
    for rt, rp in rating_pairs:
        total += (rt - rp) ** 2
    RMSE = np.sqrt(total / float(n))
    return RMSE


def computeMAE(rating_pairs):
    n = rating_pairs.shape[0]
    total = 0
    for rt, rp in rating_pairs:
        total += abs(rt - rp)
    MAE = total / float(n)
    return MAE
