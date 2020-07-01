## pypma

Python implementations for PMA algorithms based on Witten et al, 2009 and the corresponding R package PMA. Includes:

* Penalized Matrix Decomposition (PMD), which is a version of Singular Value Decomposition (X = UDV^T), where U and V can be penalized to be sparse.

* A dedicated efficient algorithm for computing Sparse CCA (that is equivalent to computing PMD with X.T @ Z, where X and Z are two observation matrices with same amount of observations).

### Installation

python setup.py install

### Usage

See examples.

### Acknowledgements

Great thanks to the original authors, see https://cran.r-project.org/web/packages/PMA/index.html
