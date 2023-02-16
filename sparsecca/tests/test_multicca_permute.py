from pprint import pprint
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .._multicca_pmd_permute import multicca_permute


def test_multicca_permute():
    np.random.seed(42)
    datasets = [np.random.rand(3, 2), np.random.rand(3, 5)]
    print(datasets)

    res = multicca_permute(datasets, penalties=np.sqrt(3) / 2)
    pprint(res)


def test_compare_multicca_permute_to_r():
    datasets = [
        pd.read_csv("sparsecca/tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("sparsecca/tests/data/multicca2.csv", sep=",", index_col=0).values,
    ]

    # R PMA::MultiCCA.permute output for same data
    cors = np.array(
        [
            0.4986115,
            0.4986115,
            0.4986115,
            0.4986115,
            0.4986115,
            0.4948202,
            0.4809967,
            0.4865116,
            0.4821666,
            0.4748496,
        ]
    )
    penalties = np.array(
        [
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.100000, 1.267105, 1.441022, 1.614938, 1.788854],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.197528, 1.388044, 1.578560, 1.769076, 1.959592],
        ]
    )
    best_penalties = np.array([1.1, 1.1])
    pvals = np.array(
        [0.242, 0.242, 0.242, 0.242, 0.242, 0.293, 0.448, 0.511, 0.556, 0.599]
    )
    zstat = np.array(
        [
            0.467956119157931,
            0.467956119157931,
            0.467956119157931,
            0.467956119157931,
            0.467956119157931,
            0.365482477335275,
            0.0579137832284457,
            -0.0553037219023547,
            -0.120807884563517,
            -0.190823297491557,
        ]
    )

    res = multicca_permute(
        datasets,
        nperms=1000,
        niter=100,
        standardize=True,
    )

    pprint(res)
    rtol = 1e-6
    assert np.allclose(cors, res["cors"], rtol=rtol)
    assert np.allclose(penalties, res["penalties"], rtol=rtol)
    assert np.allclose(best_penalties, res["bestpenalties"], rtol=rtol)

    # check statistics match somewhat
    assert spearmanr(pvals, res["pvals"])[0] == 1
    assert spearmanr(zstat, res["zstat"])[0] == 1
