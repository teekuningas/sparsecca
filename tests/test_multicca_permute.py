import numpy as np
import sparsecca


def test_multicca_permute():
    np.random.seed(42)
    datasets = [np.random.rand(3, 2), np.random.rand(3, 5)]

    res = sparsecca.multicca_permute(datasets)
    print(res)
