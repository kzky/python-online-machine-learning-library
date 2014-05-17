# python-online-machine-learning-library

## Description

This is a online machine learning library for some famous classifiers including  

* Passive Agrressive Algorithm  
    ** L1-hinge loss  
    ** L2-hinge loss  
* Confidence Weighted  
    ** variance version  
* Multi-Class Multi-Label Confidence Weighted   
    ** single contraint and diagonal matrix version  
* Adaptive Regularization of Weights  
    ** dense matrix  
    ** diagonal matrix  
* Softconfidence Weighted  
    ** L1-hinge loss  
    ** L2-hinge loss  
* Softconfidence Weighted  
    ** L1-hinge loss, single contraint, and diagonal matrix  
    ** L2-hinge loss, single contraint, and diagonal matrix  
* Logistic Regression (should not be used)  
    ** solved with stochastic gradient  

## Dependency
* numpy
* (scipy)
* (scikit-learn)

## How to use
Refer to the main function and and class method, examplify.

## Note
* Now python is only applicable.
* All labeled samples to be leaned are included in memory.
* Dense matrix format (2-dimensional ndarray) are used now.
* All algorithm except for Multi-Class Multi-Label CW and SCW are for binary classification only.
* A sample is normalized such that the norm of sample is equal to 1.
* Bias, 1 is added to a sample.

## Future Plan
* Sparse matrix is used.
* Binary classifiers are extended to 1-vs-n and/or pair-wise classifiers.
* All labeled samples to be leaned are not necessarily included in memory.

