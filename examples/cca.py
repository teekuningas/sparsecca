""" Examplifies different CCA methods
"""

import numpy as np
from numpy.linalg import svd
from statsmodels.multivariate.cancorr import CanCorr

from pypma import cca
from pypma import pmd


random_state = np.random.RandomState(10)

# simulate datasets X and Z with common factor
y = np.array([1, 6, 8, 3, 9, 3, 2, 1, 1, 4, 7, 10, 15, 10, 7])
x1 = np.array([random_state.normal() for idx in range(len(y))])
x2 = np.array([random_state.normal() for idx in range(len(y))])
x3 = np.array([random_state.normal() for idx in range(len(y))])
z1 = np.array([random_state.normal() for idx in range(len(y))])
z2 = np.array([random_state.normal() for idx in range(len(y))])
z3 = np.array([random_state.normal() for idx in range(len(y))])
z4 = np.array([random_state.normal() for idx in range(len(y))])
x1, x2 = x1 + y, x2 + y
z2, z3 = z2 - y, z3 + y
X = np.array([x1, x2, x3]).T
Z = np.array([z1, z2, z3, z4]).T

X = X - np.mean(X, axis=0)
Z = Z - np.mean(Z, axis=0)

print("Use standard CCA from statsmodels")
stats_cca = CanCorr(Z, X)

print(stats_cca.corr_test().summary())
print("X weights: ")
print(stats_cca.x_cancoef)
print("Z weights: ")
print(stats_cca.y_cancoef)

print("Use dedicated CCA method from this package (no need to compute X.T @ Z)")
U, V, D = cca(X, Z, penaltyx=1.0, penaltyz=1.0, K=3, standardize=False)

for idx in range(U.shape[1]):
    x_weights = U[:, idx]
    z_weights = V[:, idx]
    corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
    print("Corrcoef for comp " + str(idx+1) + ": " + str(corrcoef))

print("X weights: ")
print(U)
print("Z weights: ")
print(V)

print("Use SVD of X.T @ Z to compute CCA (should match the dedicated cca)")
U, D, V = svd(X.T @ Z)

for idx in range(U.shape[1]):
    x_weights = U[:, idx]
    z_weights = V[idx, :]
    corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
    print("Corrcoef for comp " + str(idx+1) + ": " + str(corrcoef))

print("X weights: ")
print(U)
print("Z weights: ")
print(V.T)


print("Use PMD of X.T @ Z to compute CCA (should match the svd")
U, V, D = pmd(X.T @ Z, K=3, penaltyu=1.0, penaltyv=1.0, standardize=False)

for idx in range(U.shape[1]):
    x_weights = U[:, idx]
    z_weights = V[:, idx]
    corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
    print("Corrcoef for comp " + str(idx+1) + ": " + str(corrcoef))

print("X weights: ")
print(U)
print("Z weights: ")
print(V)

print("Use PMD of X.T @ Z to compute CCA (with penalty)")
U, V, D = pmd(X.T @ Z, K=3, penaltyu=0.8, penaltyv=0.8, standardize=False)

for idx in range(U.shape[1]):
    x_weights = U[:, idx]
    z_weights = V[:, idx]
    corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
    print("Corrcoef for comp " + str(idx+1) + ": " + str(corrcoef))

print("X weights: ")
print(U)
print("Z weights: ")
print(V)

