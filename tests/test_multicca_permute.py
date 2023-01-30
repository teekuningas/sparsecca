from pprint import pprint
import numpy as np
import pandas as pd
import sparsecca


def test_multicca_permute():
    np.random.seed(42)
    datasets = [np.random.rand(3, 2), np.random.rand(3, 5)]

    res = sparsecca.multicca_permute(datasets)
    print(res)


def test_compare_multicca_permute_to_r():
    datasets = [
        pd.read_csv('tests/data/multicca1.csv', sep=',', index_col=0).values,
        pd.read_csv('tests/data/multicca2.csv', sep=',', index_col=0).values
    ]

    # R PMA::MultiCCA.permute output for same data
    cors = np.array(
        [0.4986115, 0.4986115, 0.4986115, 0.4986115, 0.4986115,
         0.49482022, 0.48099667, 0.48651162, 0.48216657, 0.47484963]
    )
    penalties = np.array([
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.26710519,
            1.44102159, 1.61493798, 1.78885438],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.19752832, 1.38804419,
            1.57856006, 1.76907593, 1.95959179]
    ])

    res = sparsecca.multicca_permute(
        datasets,
        nperms=10,
        niter=25,
        standardize=True,
    )

    pprint(res)
    rtol=1e-9
    assert np.allclose(cors, res['cors'], rtol=rtol)
    assert np.allclose(penalties, res['penalties'], rtol=rtol)
