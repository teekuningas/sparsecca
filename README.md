## sparsecca

Python implementations for Sparse CCA algorithms. Includes:

* Sparse (multiple) CCA based on Penalized Matrix Decomposition (PMD) from Witten et al, 2009.
* Sparse CCA based on Iterative Penalized Least Squares from Mai et al, 2019.  
  
One main difference between these two is that while the first is very simple it assumes datasets to be white.

### Installation

#### Dependencies

In addition to basic scientific packages such as numpy and scipy, iterative penalized least squares needs either glmnet\_python or pyglmnet to be installed.

#### This package can be installed normally with

```
git clone https://github.com/theislab/sparsecca  
cd sparsecca  
pip install .
```

### Usage

See examples, https://teekuningas.github.io/sparsecca

### Acknowledgements

Great thanks to the original authors, see Witten et al, 2009 and Mai et al, 2019.
