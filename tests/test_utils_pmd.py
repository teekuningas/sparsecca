from sparsecca import l2n, soft, scale, binary_search

import numpy as np
from rpy2 import robjects
from rpy2.robjects.numpy2ri import numpy2rpy


def test_l2n():
    r_l2n = robjects.r(
        """
        function(vec) {
            a <- sqrt(sum(vec^2))
            if(a==0) a <- .05
            return(a)
        }
        """
    )

    vec_a = np.array([1, 2, 3])
    a = l2n(vec_a)
    a_r = r_l2n(numpy2rpy(vec_a))
    assert np.allclose(a, a_r, rtol=1e-10)

    vec_b = np.array([0, 0, 0])
    b = l2n(vec_b)
    b_r = r_l2n(numpy2rpy(vec_b))
    assert np.allclose(b, b_r, rtol=1e-10)


def test_binary_search():
    r_binary_search = robjects.r(
        """
        l2n <- function(vec){
            a <- sqrt(sum(vec^2))
            if(a==0) a <- .05
            return(a)
        }

        soft <- function(x,d){
            return(sign(x)*pmax(0, abs(x)-d))
        }

        function(argu, sumabs) {
            if(l2n(argu)==0 | sum(abs(argu/l2n(argu))) <= sumabs) return(0)
            lam1 <- 0
            lam2 <- max(abs(argu)) - 1e-5
            for(idx in 1:150) {
                su <- soft(argu, (lam1+lam2)/2)
                if(sum(abs(su/l2n(su))) < sumabs) {
                    lam2 <- (lam1+lam2)/2
                } else {
                    lam1 <- (lam1+lam2)/2
                }
                if(lam2-lam1 < 1e-6) {
                    return((lam1+lam2)/2)
                }
            }
            print("Warning. Binary search did not quite converge..")
            return((lam1+lam2)/2)
        }
        """
    )

    a_argu = np.array([0.1, 0.2, 0.6])
    a_sumabs = 0.5

    a_lam = binary_search(a_argu, a_sumabs)
    a_lam_r = r_binary_search(numpy2rpy(a_argu), a_sumabs)

    assert np.allclose(a_lam, a_lam_r, rtol=1e-10)

    b_argu = np.array([-0.1, 0.2, -0.6])
    b_sumabs = 0.5

    b_lam = binary_search(b_argu, b_sumabs)
    b_lam_r = r_binary_search(numpy2rpy(b_argu), b_sumabs)

    assert np.allclose(b_lam, b_lam_r, rtol=1e-10)


def test_soft():
    r_soft = robjects.r(
        """
        function(x, d) {
            return(sign(x)*pmax(0, abs(x)-d))
        }
        """
    )

    x = np.array([1, 2, 3])
    d = 0.5

    x_soft = soft(x, d)
    x_soft_r = r_soft(numpy2rpy(x), d)

    assert np.allclose(x_soft, x_soft_r, rtol=1e-10)

    y = np.array([-1, -2, -3])
    y_soft = soft(y, d)

    assert np.allclose(y_soft, -x_soft, rtol=1e-10)

    z = np.array([0, 0, 0])
    z_soft = soft(z, d)

    assert np.allclose(z_soft, z, rtol=1e-10)


def test_scale():
    X = np.array([[1, 2], [3, 4]])
    X_scaled = scale(X)

    r_scale = robjects.r(
        """
        function(X) {
            X_scaled <- scale(X)
            return(X_scaled)
        }
        """
    )

    X_scaled_r = r_scale(numpy2rpy(X))

    assert np.allclose(X_scaled, X_scaled_r, rtol=1e-10)
