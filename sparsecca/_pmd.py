""" Here lies PMD as implemented in Witten et al. 2009
"""
import numpy as np
from scipy.linalg import svd

from ._utils_pmd import soft
from ._utils_pmd import l2n
from ._utils_pmd import binary_search


def smd(x_, sumabsu, sumabsv, niter, v_):
    """ computes single factor
    """
    vold = np.random.normal(size=len(v_))
    for idx in range(niter):

        if sum(abs(vold-v_)) < 1e-7:
            break

        # update u
        argu = x_ @ v_
        lamu = binary_search(argu, sumabsu)
        su = soft(argu, lamu)
        u_ = su/l2n(su)

        # update v
        vold = v_
        argv = u_.T @ x_
        lamv = binary_search(argv, sumabsv)
        sv = soft(argv, lamv)
        v_ = sv/l2n(sv)

    d_ = u_.T @ (x_ @ v_)
    return u_, v_, d_


def pmd_l1l1(x_, penaltyu=0.9, penaltyv=0.9, niter=20, K=1,
             standardize=True):
    """ pmd with l1-regularisation for both u and v
    """
    if standardize:
        x_ = x_ - np.mean(x_, axis=0)

    if penaltyu < 0 or penaltyu > 1:
        raise Exception('Penaltyu must be be between 0 and 1')

    if penaltyv < 0 or penaltyv > 1:
        raise Exception('Penaltyv must be be between 0 and 1')


    # convert penalties that lie between 0 and 1 to sumabs that 
    # should be between 1 and sqrt(p) or sqrt(q)
    sumabsu = np.sqrt(x_.shape[0])*penaltyu
    sumabsv = np.sqrt(x_.shape[1])*penaltyv

    # initialize v
    v_ = svd(x_)[2][0:K].T

    us, vs, ds = [], [], []

    xres = x_

    for k in range(K):
        rs = smd(xres, sumabsu=sumabsu, sumabsv=sumabsv, niter=niter, 
                 v_=v_[:, k])
        
        ds.append(rs[2])
        us.append(rs[0])
        vs.append(rs[1])

        xres = xres - rs[2] * (rs[0][:, np.newaxis] @ rs[1][np.newaxis, :])

    U = np.array(us).T
    V = np.array(vs).T
    D = np.diag(ds)

    return U, V, D

def pmd(x_, penaltyu=0.9, penaltyv=0.9, niter=20, K=1,
        standardize=True):
    """ computes penalized version of singular value decomposition
    """
    results = pmd_l1l1(x_, penaltyu=penaltyu, penaltyv=penaltyv, 
        niter=niter, K=K, standardize=standardize)
    return results

