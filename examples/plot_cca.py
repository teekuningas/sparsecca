""" 
Different CCA methods
=====================

Exempliefies different CCA methods

"""

# %%
# Import necessary libraries.

import pandas as pd
import numpy as np
from numpy.linalg import svd
from statsmodels.multivariate.cancorr import CanCorr

from sparsecca import cca_ipls
from sparsecca import cca_pmd
from sparsecca import multicca_pmd
from sparsecca import pmd

# %% 
# Get toy data example from seaborn

path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(path)
df = df.dropna()

X = df[['bill_length_mm', 'bill_depth_mm']]
Z = df[['flipper_length_mm', 'body_mass_g']]

X = ((X - np.mean(X)) / np.std(X)).to_numpy()
Z = ((Z - np.mean(Z)) / np.std(Z)).to_numpy()

# %%
# Define function for printing weights
def print_weights(name, weights):
    # first = weights[:, 0] / np.max(np.abs(weights[:, 0]))
    first = weights[:, 0]
    print(name + ': ' + ', '.join(['{:.3f}'.format(item) for item in first]))

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

U, V, D = pmd(X.T @ Z, K=2, penaltyu=0.8, penaltyv=0.9, standardize=False)

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

