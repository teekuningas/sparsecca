import numpy as np
from scipy.linalg import svd

from ._utils_pmd import soft
from ._utils_pmd import l2n
from ._utils_pmd import binary_search


def get_crit(datasets, ws):
    crit = 0
    for ii in range(1, len(datasets)):
        for jj in range(0, ii):
            crit = crit + ws[ii].T @ datasets[ii].T @ datasets[jj] @ ws[jj]
    return crit


def update_w(datasets, idx, sumabs, ws, ws_final):
    tots = 0
    for jj in [ii for ii in range(len(datasets)) if ii != idx]:
        diagmat = ws_final[idx].T @ datasets[idx].T @ datasets[jj] @ ws_final[jj]
        for a in range(diagmat.shape[0]):
            for b in range(diagmat.shape[1]):
                if a != b:
                    diagmat[a, b] = 0

        tots = (tots + datasets[idx].T @ datasets[jj] @ ws[jj] -
                ws_final[idx] @ diagmat @ ws_final[jj].T @ ws[jj])

        sumabs = binary_search(tots, sumabs)
        w_ = soft(tots, sumabs) / l2n(soft(tots, sumabs))

        return w_


def multicca(datasets, penalties, niter=25, K=1, standardize=True):
    """
    """
    for data in datasets:
        if data.shape[1] < 2:
            raise Exception('Need at least 2 features in each datset')

    if standardize:
        for idx in range(len(datasets)):
            datasets[idx] = datasets[idx] - np.mean(datasets[idx], axis=0)

    ws = []
    for idx in range(len(datasets)):
        ws.append(svd(datasets[idx])[2][0:K].T)

    sumabs = []
    for idx, penalty in enumerate(penalties):
        sumabs.append(penalty*np.sqrt(datasets[idx].shape[1]))

    ws_init = ws

    ws_final = []
    for idx in range(len(datasets)):
        ws_final.append(np.zeros((datasets[idx].shape[1], K)))

    for comp_idx in range(K):
        ws = []
        for idx in range(len(ws_init)):
            ws.append(ws_init[idx][:, comp_idx])

        curiter = 0
        crit_old = -10
        crit = -20
        storecrits = []

        while (curiter < niter and 
               np.abs(crit_old - crit) / np.abs(crit_old) > 0.001 and
               crit_old != 0):
            crit_old = crit
            crit = get_crit(datasets, ws)

            storecrits.append(crit)
            curiter += 1
            for idx in range(len(datasets)):
                ws[idx] = update_w(datasets, idx, sumabs[idx], ws, ws_final)

        for idx in range((len(datasets))):
            ws_final[idx][:, comp_idx] = ws[idx]

    return ws_final

