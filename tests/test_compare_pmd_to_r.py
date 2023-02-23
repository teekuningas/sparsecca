import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

import numpy as np

from sparsecca import cca_pmd


def test_compare_pmd_to_r():
    """Compares the output of the original R implementation to the implementation
    of this python implementation using example data.

    Thanks for JohannesWiesner for R and python code.
    """

    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled("PMA"):
        utils.install_packages("PMA", verbose=True)

    robjects.r(
        """

        ## Run example from PMA package ################################################
    
        library(PMA)
    
        # first, do CCA with type="standard"
        # A simple simulated example
        set.seed(3189)
        u <- matrix(c(rep(1,25),rep(0,75)),ncol=1)
        v1 <- matrix(c(rep(1,50),rep(0,450)),ncol=1)
        v2 <- matrix(c(rep(0,50),rep(1,50),rep(0,900)),ncol=1)
        x <- u%*%t(v1) + matrix(rnorm(100*500),ncol=500)
        z <- u%*%t(v2) + matrix(rnorm(100*1000),ncol=1000)
    
        # Can run CCA with default settings, and can get e.g. 3 components
        out <- CCA(x,z,typex="standard",typez="standard",K=3,penaltyx=0.3,penaltyz=0.3)

        out_u = out$u
        out_v = out$v
    
    """
    )

    # Get the r data as numpy arrays
    out_u = np.array(robjects.globalenv["out_u"])
    out_v = np.array(robjects.globalenv["out_v"])
    x = np.array(robjects.globalenv["x"])
    z = np.array(robjects.globalenv["z"])

    # Compute cca with the same data as in
    u, v, d = cca_pmd(x, z, penaltyx=0.3, penaltyz=0.3, K=3, niter=15)

    assert np.allclose(np.abs(u), np.abs(out_u))
    assert np.allclose(np.abs(v), np.abs(out_v))
