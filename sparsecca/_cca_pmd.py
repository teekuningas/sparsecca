""" Here lies CCA as implemented in Witten et al. 2009
"""
import numpy as np
from scipy.linalg import svd

from ._utils_pmd import soft
from ._utils_pmd import l2n
from ._utils_pmd import binary_search


def sparse_cca(x_, z_, v_, penaltyx, penaltyz, niter):
    """ computes one canonical component """

    vold = np.random.normal(size=len(v_))
    u_ = np.random.normal(size=x_.shape[1])
    for idx in range(niter):
        if np.sum(np.isnan(u_)) > 0 or np.sum(np.isnan(v_)) > 0:
            v_ = np.array([0]*len(v_))
            vold = v_

        if np.sum(np.abs(vold-v_)) < 1e-6:
            break

        # update u
        argu = (z_ @ v_) @ x_
        lamu = binary_search(argu, penaltyx*np.sqrt(x_.shape[1]))
        su = soft(argu, lamu)
        u_ = su / l2n(su)

        # update v
        vold = v_
        argv = (x_ @ u_) @ z_
        lamv = binary_search(argv, penaltyz*np.sqrt(z_.shape[1]))
        sv = soft(argv, lamv)
        v_ = sv / l2n(sv)

    d_ = np.sum((x_ @ u_) * (z_ @ v_))

    if np.sum(np.isnan(u_)) > 0 or np.sum(np.isnan(v_)) > 0:
        u_ = np.array([0]*x_.shape[1])
        v_ = np.array([0]*z_.shape[1])
        d_ = 0
  
    return u_, v_, d_


def cca_algorithm(x_, z_, v_, penaltyx, penaltyz, K, niter):
    """ implements the biconvex optimization algorithm for finding u and v
    """
    us, vs, ds = [], [], []

    xres = x_
    zres = z_
    
    for k in range(K):
        results = sparse_cca(xres, zres, v_[:, k], penaltyx, penaltyz,
                             niter)

        coef = results[2]
        ds.append(coef)

        xres = np.vstack([xres, np.sqrt(coef)*results[0]])
        zres = np.vstack([zres, -np.sqrt(coef)*results[1]])

        us.append(results[0])
        vs.append(results[1])

    U = np.array(us).T
    V = np.array(vs).T
    D = np.diag(ds)

    return U, V, D


def cca(x_, z_, penaltyx=0.9, penaltyz=0.9, K=1, niter=20, 
        standardize=True):
    """ given x and z datasets, computes canonical weights
    """
    if x_.shape[1] < 2:
        raise Exception('Need at least two features in dataset x')
    if z_.shape[1] < 2:
        raise Exception('Need at least two features in dataset y')
    if not x_.shape[0] == z_.shape[0]:
        raise Exception('x and z must have same number of rows')

    if standardize:
        x_ = (x_ - np.mean(x_, axis=0)[np.newaxis, :]) / np.std(x_, axis=0, ddof=1)[np.newaxis, :]
        z_ = (z_ - np.mean(z_, axis=0)[np.newaxis, :]) / np.std(z_, axis=0, ddof=1)[np.newaxis, :]

    if np.abs(np.mean(x_)) > 1e-10 or np.abs(np.mean(z_)) > 1e-10:
        print("Warning, cca was run without first subtracting "
              "out the means of the data matrices.")

    if penaltyx < 0 or penaltyx > 1:
        raise Exception('Penaltyx must be be between 0 and 1')

    if penaltyz < 0 or penaltyz > 1:
        raise Exception('Penaltyz must be be between 0 and 1')

    # find initial value with standard SVD
    v_ = svd((x_.T @ z_))[2][0:K].T

    result = cca_algorithm(x_, z_, v_, penaltyx, penaltyz, K, niter)
    return result

