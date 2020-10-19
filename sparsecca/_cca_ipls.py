""" Here lies scca based on Mai et al, 2019
"""
import numpy as np
from scipy.linalg import svd


def init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, n_pairs):
    """ supports only initialization by svd """
    results = svd(sigma_YX_hat)
    alpha_init = results[0][:, :n_pairs]
    beta_init = (results[2].T)[:, :n_pairs]

    alpha_scale = np.diag(alpha_init.T @ sigma_Y_hat @ alpha_init)
    beta_scale = np.diag(beta_init.T @ sigma_X_hat @ beta_init)

    for idx in range(n_pairs):
        alpha_init[:, idx] = alpha_init[:, idx] / np.sqrt(alpha_scale)[idx]
        beta_init[:, idx] = beta_init[:, idx] / np.sqrt(beta_scale)[idx]

    return alpha_init, beta_init


def find_omega(sigma_YX_hat, idx_pairs,
               alpha, beta, x, y):
    """
    """
    n_rows = x.shape[0]
    if idx_pairs > 0:
        rho = alpha.T @ sigma_YX_hat @ beta
        omega = np.diag([1]*n_rows) - (y @ alpha @ rho @ beta.T @ (x.T / n_rows))
    else:
        omega = np.diag([1]*n_rows)
    return omega


def scca_solution(x, y, x_omega, y_omega, alpha0, beta0, 
                  alpha_lambda_ratio, beta_lambda_ratio, 
                  alpha_lambda, beta_lambda, niter, eps, glm_impl):
    """ computes one pair of canonical weights
    """
    for idx in range(niter):
        x0 = x_omega @ beta0

        if glm_impl == 'glmnet_python':
            from glmnet_python import glmnet
            lambda_a = np.array([alpha_lambda])
            m_ = glmnet(x=y.copy(), y=x0.copy(), standardize=False, intr=False, 
                        family='gaussian', lambdau=lambda_a, 
                        alpha=alpha_lambda_ratio)
            alpha1 = m_['beta'][:, -1]
        elif glm_impl == 'pyglmnet':
            from pyglmnet import GLM
            alpha1 = GLM(distr='gaussian', alpha=alpha_lambda_ratio, reg_lambda=alpha_lambda,
                         fit_intercept=False).fit(y.copy(), x0.copy()).beta_
        else:
            raise Exception(str(glm_impl) + ' not supported.')

        if np.sum(np.abs(alpha1)) < eps: 
            alpha0 = [0]*y.shape[1]
            break

        idx_nz = np.where(alpha1 != 0)[0]
        alpha1_scale = y[:, idx_nz] @ alpha1[idx_nz]

        alpha1 = alpha1 / np.sqrt(alpha1_scale @ alpha1_scale / (x.shape[0] - 1))

        y0 = y_omega @ alpha1

        if glm_impl == 'glmnet_python':
            from glmnet_python import glmnet
            lambda_b = np.array([beta_lambda])
            m_ = glmnet(x=x.copy(), y=y0.copy(), standardize=False, intr=False, 
                        family='gaussian', lambdau=lambda_b,
                        alpha=beta_lambda_ratio)
            beta1 = m_['beta'][:, -1]
        elif glm_impl == 'pyglmnet':
            from pyglmnet import GLM
            beta1 = GLM(distr='gaussian', alpha=beta_lambda_ratio, reg_lambda=beta_lambda,
                         fit_intercept=False).fit(x.copy(), y0.copy()).beta_
        else:
            raise Exception(str(glm_impl) + ' not supported.')



        if np.sum(np.abs(beta1)) < eps:
            beta0 = [0]*x.shape[1]
            break

        idx_nz = np.where(beta1 != 0)[0]
        beta1_scale = x[:, idx_nz] @ beta1[idx_nz]
        beta1 = beta1 / np.sqrt(beta1_scale @ beta1_scale / (x.shape[0] - 1))

        if (np.sum(np.abs(alpha1 - alpha0)) < eps and 
                np.sum(np.abs(beta1 - beta0)) < eps):
            break

        alpha0 = alpha1
        beta0 = beta1

    return alpha0, beta0


def scca(x, y, alpha_lambda_ratio=1.0, beta_lambda_ratio=1.0,
         alpha_lambda=0.05, beta_lambda=0.05, niter=100, n_pairs=1, 
         standardize=True, eps=1e-4, glm_impl='pyglmnet'):
    """ compute penalized canonical weights for x (n_obs, n_features) and 
    y (n_obs, n_features).
    """

    x = x - np.mean(x, axis=0)
    y = y - np.mean(y, axis=0)

    if standardize:
        x = x / np.std(x, axis=0, ddof=1)
        y = y / np.std(y, axis=0, ddof=1)

    sigma_YX_hat = (y.T @ x) / (x.shape[0] - 1)
    sigma_X_hat = (x.T @ x) / (x.shape[0] - 1)
    sigma_Y_hat = (y.T @ y) / (y.shape[0] - 1)

    alpha = np.zeros((y.shape[1], n_pairs))
    beta = np.zeros((x.shape[1], n_pairs))

    alpha_init, beta_init = init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, 
                                  n_pairs)

    for idx_pairs in range(n_pairs):

        omega = find_omega(sigma_YX_hat, idx_pairs,
            alpha=alpha[:, :idx_pairs],
            beta=beta[:, :idx_pairs],
            x=x, y=y)

        x_tmp = omega @ x
        y_tmp = omega.T @ y

        alpha0 = alpha_init[:, idx_pairs]
        beta0 = beta_init[:, idx_pairs]

        results = scca_solution(x=x, y=y, x_omega=x_tmp, y_omega=y_tmp,
                                alpha0=alpha0, beta0=beta0,
                                alpha_lambda_ratio=alpha_lambda_ratio,
                                beta_lambda_ratio=beta_lambda_ratio,
                                alpha_lambda=alpha_lambda, 
                                beta_lambda=beta_lambda,
                                niter=niter, eps=eps,
                                glm_impl=glm_impl)

        alpha[:, idx_pairs] = results[0]
        beta[:, idx_pairs] = results[1]

    return beta, alpha

