import numpy as np
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.packages as rpackages

from sparsecca import multicca_pmd


def test_compare_multicca_to_r_2datasets():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0).values,
    ]

    ws, _ = multicca_pmd(datasets, [1.5, 1.5], K=3, standardize=True, niter=25)

    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled("PMA"):
        utils.install_packages("PMA", verbose=True)

    r_pma_ws = robjects.r(
        """
        library("PMA")

        cls <- c(lat = "numeric", lon = "numeric")
        data1 <- read.table("tests/data/multicca1.csv", sep = ",", header = TRUE)
        rownames(data1) <- data1$X
        data1 <- data1[, 2:ncol(data1)]

        data2 <- read.table("tests/data/multicca2.csv", sep = ",", header = TRUE)
        rownames(data2) <- data2$X
        data2 <- data2[, 2:ncol(data2)]

        datasets <- list(data1, data2)
        res <- MultiCCA(
            datasets,
            type = "standard",
            penalty = 1.5,
            ncomponents = 3,
            standardize = TRUE
        )

        res$ws
        """
    )

    for i in range(len(r_pma_ws)):
        assert np.allclose(ws[i], np.array(r_pma_ws[i]), rtol=1e-10)


def test_compare_multicca_to_r_3datasets():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
    ]

    ws, _ = multicca_pmd(datasets, [1.5, 1.5, 1.5], K=3, standardize=True, niter=25)

    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled("PMA"):
        utils.install_packages("PMA", verbose=True)

    r_pma_ws = robjects.r(
        """
        library("PMA")

        cls <- c(lat = "numeric", lon = "numeric")
        data1 <- read.table("tests/data/multicca1.csv", sep = ",", header = TRUE)
        rownames(data1) <- data1$X
        data1 <- data1[, 2:ncol(data1)]

        data2 <- read.table("tests/data/multicca2.csv", sep = ",", header = TRUE)
        rownames(data2) <- data2$X
        data2 <- data2[, 2:ncol(data2)]

        data3 <- read.table("tests/data/multicca1.csv", sep = ",", header = TRUE)
        rownames(data3) <- data3$X
        data3 <- data3[, 2:ncol(data3)]

        datasets <- list(data1, data2, data3)
        res <- MultiCCA(
            datasets,
            type = "standard",
            penalty = 1.5,
            ncomponents = 3,
            standardize = TRUE
        )

        res$ws
        """
    )

    print("\n\n===== OUTPUTS: =====")

    print("\nsparsecca.multicca_pmd")
    for i, w in enumerate(ws):
        print(f"ws[{i}]:")
        print(w)

    print("\nPMA.MultiCCA")
    for i, w in enumerate(r_pma_ws):
        print(f"ws[{i}]:")
        print(np.array(w))

    for i in range(len(r_pma_ws)):
        assert np.allclose(ws[i], np.array(r_pma_ws[i]), rtol=1e-10)


if __name__ == "__main__":
    test_compare_multicca_to_r_2datasets()
