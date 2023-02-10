import copy
import numpy as np
from ._multicca_pmd import multicca
from ._utils_pmd import scale


def multicca_permute(
        datasets: list,
        penalties: int = None,
        nperms: int = 10,
        niter: int = 3,
        standardize: bool = True,
):
    """
    Select tuning parameters for sparse multiple canonical correlation analysis
    using the penalized matrix decomposition.

     Params
    ------
    datasets: list of datasets
    niter : int (default: 3)
    standardize : bool (default: True)
        Whether to center and scale each dataset before computing sparse
        canonical variates.
    """
    datasets = copy.deepcopy(datasets)
    K = len(datasets)
    for k in range(K):
        if datasets[k].shape[0] < 2:
            raise ValueError('Need at least 2 features in each data set!')
        if standardize:
            datasets[k] = scale(datasets[k], center=True, scale=True)

    if penalties is None:
        # only "standard" mode
        penalties = np.zeros(shape=(K, 10))
        for k in range(K):
            n_cols = datasets[k].shape[1]
            vals = np.linspace(start=.1, stop=.8, num=10) * np.sqrt(n_cols)
            penalties[k, :] = [np.max([v, 1.1]) for v in vals]
    else:
        penalties = np.full(shape=(K, 1), fill_value=penalties)

    penalty_columns = penalties.shape[1]
    cors = np.zeros(shape=penalty_columns)
    for i in range(penalty_columns):
        ws, ws_init = multicca(datasets, penalties=penalties[:, i], niter=niter)
        cors[i] = get_cors(datasets, ws)

    # permute
    perm_cors = np.zeros(shape=(nperms, penalty_columns))
    for j in range(nperms):
        datasets_perm = datasets

        # permute the data
        for k in range(K):
            datasets_perm[k] = np.random.permutation(datasets_perm[k])

        # run MultiCCA
        for i in range(penalty_columns):
            ws, _ = multicca(datasets, penalties=penalties[:, i], niter=niter)
            perm_cors[j, i] = get_cors(datasets, ws)

    # get summary statistics
    p_vals = []
    z_stats = []
    for i in range(penalty_columns):
        cor = cors[i]
        perm_cor = perm_cors[:, i]

        p_value = np.mean(perm_cor >= cor)
        p_vals.append(p_value)

        z_stat = (cor - np.mean(perm_cor)) / (np.std(perm_cor) + .05)
        z_stats.append(z_stat)

    return dict(
        pvals=np.array(p_vals),
        zstat=np.array(z_stats),
        bestpenalties=penalties[:, np.argmax(z_stats)],
        cors=np.array(cors),
        penalties=penalties,
        nperms=nperms
    )


def get_cors(datasets, ws):
    """
    Get sum of pairwise correlations
    """
    K = len(datasets)
    cors = []
    for i in range(1, K):
        for j in range(i):
            x = np.ravel(datasets[i] @ ws[i])
            y = np.ravel(datasets[j] @ ws[j])
            cor = np.corrcoef(x, y)
            cors.append(cor[0, 1])
    return sum(cors)
