""" Here lies CCA as implemented in Witten et al. 2009
"""
import numpy as np
from scipy.linalg import svd


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
    """
    """
    U, V, D = [], [], []

    xres = x_
    zres = z_
    
    for k in range(K):
        results = sparse_cca(xres, zres, v_[:, k], penaltyx, penaltyz,
                             niter)

        coef = results[2]
        D.append(coef)

        xres = np.vstack([xres, np.sqrt(coef)*results[0]])
        zres = np.vstack([zres, -np.sqrt(coef)*results[1]])

        U.append(results[0])
        V.append(results[1])

    U = np.array(U)
    V = np.array(V)

    return U, V, D


def cca(x_, z_, penaltyx=0.9, penaltyz=0.9, K=1, niter=15, 
        standardize=True):
    """
    """
    if x_.shape[1] < 2:
        raise Exception('Need at least two features in dataset x')
    if z_.shape[1] < 2:
        raise Exception('Need at least two features in dataset y')
    if not x_.shape[0] == z_.shape[0]:
        raise Exception('x and z must have same number of rows')

    if standardize:
        x_ = x_ - np.mean(x_, axis=0)
        z_ = z_ - np.mean(z_, axis=0)

    if penaltyx < 0 or penaltyx > 1:
        raise Exception('Penaltyx must be be between 0 and 1')

    if penaltyz < 0 or penaltyz > 1:
        raise Exception('Penaltyz must be be between 0 and 1')

    # find initial value with standard SVD
    v_ = svd((x_.T @ z_))[2][0:K].T

    result = cca_algorithm(x_, z_, v_, penaltyx, penaltyz, K, niter)
    return result


if __name__ == '__main__':
    """
    """
    random_state = np.random.RandomState(20)
    
    # simulate datasets X and Z with common factor
    y = np.array([1, 6, 8, 3, 9, 3, 2, 1, 1, 4, 7, 10, 15, 10, 7])
    x1 = np.array([random_state.normal() for idx in range(len(y))])
    x2 = np.array([random_state.normal() for idx in range(len(y))])
    x3 = np.array([random_state.normal() for idx in range(len(y))])
    z1 = np.array([random_state.normal() for idx in range(len(y))])
    z2 = np.array([random_state.normal() for idx in range(len(y))])
    z3 = np.array([random_state.normal() for idx in range(len(y))])
    z4 = np.array([random_state.normal() for idx in range(len(y))])
    x1, x2 = x1 + y, x2 + y
    z2, z3 = z2 - y, z3 + y
    X = np.array([x1, x2, x3])
    Z = np.array([z1, z2, z3, z4])

    print("Use standard CCA from statsmodels")
    from statsmodels.multivariate.cancorr import CanCorr
    print("statsmodels CCA: ")
    stats_cca = CanCorr(Z.T, X.T)

    print(stats_cca.corr_test().summary())
    print("X weights: ")
    print(stats_cca.x_cancoef)
    print("Z weights: ")
    print(stats_cca.y_cancoef)

    print("Use CCA by PMD")
    results = cca(X.T, Z.T, penaltyx=1.0, penaltyz=1.0, K=3, standardize=True)

    for idx in range(len(results[2])):
        x_weights = results[0][idx]
        z_weights = results[1][idx]
        corrcoef = np.corrcoef(np.dot(x_weights, X), np.dot(z_weights, Z))[0, 1]
        print("Corrcoef for comp " + str(idx+1) + ": " + str(corrcoef))

    print("X weights: ")
    print(results[0].T)
    print("Z weights: ")
    print(results[1].T)

