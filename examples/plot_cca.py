""" 
Different CCA methods
=====================

Exempliefies different CCA methods

"""


# %%
# Import necessary libraries.

import numpy as np
from numpy.linalg import svd
from statsmodels.multivariate.cancorr import CanCorr

from sparsecca import cca_ipls
from sparsecca import cca_pmd
from sparsecca import multicca_pmd
from sparsecca import pmd

# %% 
# Simulate correlated datasets so that 1st and 2nd variable of X dataset are correlated with 2nd, 3rd and 4th variables of the Z dataset.

# For consistency
rand_state = np.random.RandomState(15)

# Simulate correlated datasets
u_ = np.concatenate([np.ones(125), np.zeros(375)])
v1 = np.concatenate([np.ones(2), np.zeros(4)])
v2 = np.concatenate([np.zeros(1), np.ones(3), np.zeros(1)])
X = u_[:, np.newaxis] @ v1[np.newaxis, :] + rand_state.randn(500*6).reshape(500, 6)
Z = u_[:, np.newaxis] @ v2[np.newaxis, :] + rand_state.randn(500*5).reshape(500, 5)

# standardize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)

# %%
# Define function for printing weights
def print_weights(name, weights):
    first = weights[:, 0]
    print(name + ': ' + ', '.join(['{:.3f}'.format(item) for item in first / np.max(first)]))

# %%
# First, let's try CanCorr function from statsmodels package.

stats_cca = CanCorr(Z, X)

print(stats_cca.corr_test().summary())
print_weights('X', stats_cca.x_cancoef)
print_weights('Z', stats_cca.y_cancoef)

# %% 
# Next, use CCA algorithm from Witten et al.

U, V, D = cca_pmd(X, Z, penaltyx=1.0, penaltyz=1.0, K=2, standardize=False)

x_weights = U[:, 0]
z_weights = V[:, 0]
corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
print("Corrcoef for comp 1: " + str(corrcoef))

print_weights('X', U)
print_weights('Z', V)

# %%
# As the CCA algorithm in Witten et al is faster version of 
# computing SVD of X.T @ Z, try that.

U, D, V = svd(X.T @ Z)

x_weights = U[:, 0]
z_weights = V[0, :]
corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
print("Corrcoef for comp 1: " + str(corrcoef))

print_weights('X', U)
print_weights('V', V.T)

# %%
# The novelty in Witten et al is developing matrix decomposition similar
# to SVD, but which allows to add convex penalties (here lasso).
# Using that to X.T @ Z without penalty results to same as above.

U, V, D = pmd(X.T @ Z, K=2, penaltyu=1.0, penaltyv=1.0, standardize=False)

x_weights = U[:, 0]
z_weights = V[:, 0]
corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
print("Corrcoef for comp 1: " + str(corrcoef))

print_weights('X', U)
print_weights('Z', V)

# %%
# However, when you add penalties, you get a sparse version of CCA.

U, V, D = pmd(X.T @ Z, K=2, penaltyu=0.9, penaltyv=0.9, standardize=False)

x_weights = U[:, 0]
z_weights = V[:, 0]
corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
print("Corrcoef for comp 1: " + str(corrcoef))

print_weights('X', U)
print_weights('Z', V)

# %%
# PMD is really fantastically simple and powerful idea, and as seen, 
# can be used to implement sparse CCA. However, for SVD(X.T @ Z) to be 
# equivalent to CCA, cov(X) and cov(Z) should be diagonal,
# which can sometimes give problems. Another CCA algorithm allowing convex penalties
# that does not require cov(X) and cov(Z) to be diagonal, was presented in 
# Mai et al (2019). It is based on iterative least squares formulation, and as it is
# solved with GLM, it allows elastic net -like weighting of L1 and L2 -norms for 
# both datasets separately.

X_weights, Z_weights = cca_ipls(X, Z, alpha_lambda=0.0, beta_lambda=0.0, standardize=False,
                                n_pairs=2, glm_impl='glmnet_python')

x_weights = X_weights[:, 0]
z_weights = Z_weights[:, 0]
corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
print("Corrcoef for comp 1: " + str(corrcoef))

print_weights("X", X_weights)
print_weights("Z", Z_weights)

