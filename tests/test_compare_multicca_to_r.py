import numpy as np
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.numpy2ri import numpy2rpy

from sparsecca import multicca_pmd

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)

if not rpackages.isinstalled("PMA"):
    utils.install_packages("PMA", verbose=True)

r_multicca = robjects.r(
    """
    library("PMA")

    function (datasets) {
        res <- MultiCCA(
            datasets,
            type = "standard",
            penalty = 1.5,
            ncomponents = 3,
            standardize = TRUE
        )

        res$ws
    }
    """
)


def test_compare_multicca_to_r_2datasets_equal_feature_length():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0)
        .drop("PC_6", axis=1)
        .values,
    ]

    datasets_r = [numpy2rpy(x) for x in datasets]

    ws, _ = multicca_pmd(datasets, [1.5, 1.5], K=3, standardize=True, niter=25)
    r_ws = r_multicca(datasets_r)

    for i in range(len(r_ws)):
        assert np.allclose(ws[i], np.array(r_ws[i]), rtol=1e-10)


def test_compare_multicca_to_r_3datasets_equal_feature_length():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0)
        .drop("PC_6", axis=1)
        .values,
        pd.read_csv("tests/data/multicca3.csv", sep=",", index_col=0)
        .drop("PC_6", axis=1)
        .values,
    ]

    datasets_r = [numpy2rpy(x) for x in datasets]

    ws, _ = multicca_pmd(datasets, [1.5, 1.5, 1.5], K=3, standardize=True, niter=25)

    r_ws = r_multicca(datasets_r)

    print("\n\n===== OUTPUT CORRELATION: =====")
    corrcoefs = []

    for i in range(len(r_ws)):
        corrcoef = np.corrcoef(np.array(ws[i]).flatten(), np.array(r_ws[i]).flatten())[
            0, 1
        ]
        corrcoefs.append(corrcoef)
        print(f"corrcoef[{i}]: {corrcoef}")

    print("\n\n===== OUTPUTS: =====")

    print("\nsparsecca.multicca_pmd")
    for i, w in enumerate(ws):
        print(f"ws[{i}]:")
        print(w)

    print("\nPMA.MultiCCA")
    for i, w in enumerate(r_ws):
        print(f"ws[{i}]:")
        print(np.array(w))

    for i in range(len(r_ws)):
        assert np.allclose(ws[i], np.array(r_ws[i]), rtol=1e-10)


def test_compare_multicca_to_r_2datasets_unequal_feature_length():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0).values,
    ]

    datasets_r = [numpy2rpy(x) for x in datasets]

    ws, _ = multicca_pmd(datasets, [1.5, 1.5], K=3, standardize=True, niter=25)
    r_ws = r_multicca(datasets_r)

    for i in range(len(r_ws)):
        assert np.allclose(ws[i], np.array(r_ws[i]), rtol=1e-10)


def test_compare_multicca_to_r_3datasets_unequal_feature_length():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca3.csv", sep=",", index_col=0)
        .drop(["PC_6", "PC_5"], axis=1)
        .values,
    ]

    datasets_r = [numpy2rpy(x) for x in datasets]

    ws, _ = multicca_pmd(datasets, [1.5, 1.5, 1.5], K=3, standardize=True, niter=25)

    r_ws = r_multicca(datasets_r)

    print("\n\n===== OUTPUT CORRELATION: =====")
    corrcoefs = []

    for i in range(len(r_ws)):
        corrcoef = np.corrcoef(np.array(ws[i]).flatten(), np.array(r_ws[i]).flatten())[
            0, 1
        ]
        corrcoefs.append(corrcoef)
        print(f"corrcoef[{i}]: {corrcoef}")

    print("\n\n===== OUTPUTS: =====")

    print("\nsparsecca.multicca_pmd")
    for i, w in enumerate(ws):
        print(f"ws[{i}]:")
        print(w)

    print("\nPMA.MultiCCA")
    for i, w in enumerate(r_ws):
        print(f"ws[{i}]:")
        print(np.array(w))

    for i in range(len(r_ws)):
        assert np.allclose(ws[i], np.array(r_ws[i]), rtol=1e-10)
