import numpy as np
import pandas as pd
from sparsecca import multicca_pmd

datasets = [
    pd.read_csv('/home/icb/yuge.ji/sparsecca/tests/multicca1.csv', sep=',', index_col=0).values,
    pd.read_csv('/home/icb/yuge.ji/sparsecca/tests/multicca2.csv', sep=',', index_col=0).values
]

ws = multicca_pmd(datasets, [1.5, 1.5], K=3, standardize=True, niter=25)

# temporary while we get R working in a python env
rtol=1e-10
assert(np.isclose(ws[0][1][0], 0.86158127, rtol=rtol))
assert(np.isclose(ws[0][1][1], 0, rtol=rtol))
assert(np.isclose(ws[0][1][2], -0.55598863, rtol=rtol))
