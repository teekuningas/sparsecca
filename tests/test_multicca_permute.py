from pprint import pprint
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.numpy2ri import numpy2rpy

from sparsecca import multicca_permute

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)

if not rpackages.isinstalled("PMA"):
    utils.install_packages("PMA", verbose=True)

r_multicca_permute = robjects.r(
    """
    library("PMA")

    function (datasets) {
        res <- MultiCCA.permute(datasets, nperms = 100, niter = 10)

        return (list(
            zstat = res$zstat,
            pvals = res$pvals,
            bestpenalties = res$bestpenalties,
            penalties = res$penalties,
            cors = res$cors,
            corperms = res$corperms,
            ws.init = res$ws.init
        ))
    }
    """
)


def test_multicca_permute():
    np.random.seed(42)
    datasets = [np.random.rand(3, 2), np.random.rand(3, 5)]
    print(datasets)

    res = multicca_permute(datasets, penalties=np.sqrt(3) / 2)
    pprint(res)


def test_compare_multicca_permute_to_r():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0).values,
    ]

    datasets_r = [numpy2rpy(dataset) for dataset in datasets]

    res = multicca_permute(
        datasets,
        nperms=100,
        niter=10,
        standardize=True,
    )

    r_res = r_multicca_permute(
        datasets_r,
    )

    rtol = 1e-6

    print(r_res)

    assert np.allclose(r_res[4], res["cors"], rtol=rtol)
    assert np.allclose(r_res[3], res["penalties"], rtol=rtol)
    assert np.allclose(r_res[2], res["bestpenalties"], rtol=rtol)

    # check statistics match somewhat
    assert spearmanr(r_res[1], res["pvals"])[0] >= 0.95
    assert spearmanr(r_res[0], res["zstat"])[0] >= 0.95
