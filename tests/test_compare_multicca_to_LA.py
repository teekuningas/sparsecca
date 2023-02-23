import numpy as np
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from sparsecca._multicca_lp import lp_pmd

def test_compare_multicca_to_Linear_approach():
    datasets = [
        pd.read_csv("tests/data/multicca2.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca3.csv", sep=",", index_col=0).values,
    ]

    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled("PMA"):
        utils.install_packages("PMA", verbose=True)

    r_pma_ws = robjects.r(
        """
        library("PMA")

        cls <- c(lat = "numeric", lon = "numeric")
        data1 <- read.table("tests/data/multicca2.csv", sep = ",", header = TRUE)
        rownames(data1) <- data1$X
        data1 <- data1[, 2:ncol(data1)]

        data2 <- read.table("tests/data/multicca3.csv", sep = ",", header = TRUE)
        rownames(data2) <- data2$X
        data2 <- data2[, 2:ncol(data2)]

        datasets <- list(data1, data2)
        res <- MultiCCA(
            datasets,
            type = "standard",
            penalty = 1.5,
            ncomponents = 1,
            standardize = TRUE
        )

        res$ws
        """
    )

    ws_LA, _ = lp_pmd(datasets, [1.5, 1.5], K=1, standardize=True, mimic_R=True)

    # checking correlation between R weigths and LA weigths 
    for i in range(len(r_pma_ws)):
        print(f"\nDataset: {i+1} ")
        for k in range(len(np.array(r_pma_ws[i])[0])):
            print(f"k: {k+1}")
            print(f"R weigth:\n {np.array(r_pma_ws)[i,:,k]}")
            print(f"LA weigth: \n {ws_LA[i,:,k]}")
            print(f"correlation: \n{np.corrcoef(ws_LA[i,:,k].flatten(), np.array(r_pma_ws)[i,:,k].flatten())}")
            # assert np.allclose(ws_LA[i,:,k], np.array(r_pma_ws)[i,:,k], atol=.5)
            assert np.allclose(np.corrcoef(ws_LA[i,:,k].flatten(), np.array(r_pma_ws)[i,:,k].flatten()), 1, atol=.3)
