""" Helper functions
"""
import numpy as np


def l2n(vec):
    """ computes "safe" l2 norm """
    norm = np.sqrt(np.sum(vec**2))
    if norm == 0:
        norm = 0.05
    return norm


def binary_search(argu, sumabs):
    """ 
    """
    if l2n(argu) == 0 or np.sum(np.abs(argu/l2n(argu))) <= sumabs:
        return 0 

    lam1 = 0
    lam2 = np.max(np.abs(argu)) - 1e-5

    for idx in range(150):
        su = soft(argu, (lam1 + lam2) / 2)
        if np.sum(np.abs(su/l2n(su))) < sumabs:
            lam2 = (lam1 + lam2) / 2
        else:
            lam1 = (lam1 + lam2) / 2
        if lam2 - lam1 < 1e-6:
            return (lam1 + lam2) / 2

    print("Warning. Binary search did not quite converge..")
    return (lam1 + lam2) / 2


def soft(x_, d_):
    """ soft-thresholding operator """
    return np.sign(x_) * (np.abs(x_)-d_).clip(min=0)

