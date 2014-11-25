# python-online-machine-learning-library

## Description

This is a python online machine learning library (POMLL) for some famous classifiers.  
The following classifiers are implemented.

* Passive Agrressive Algorithm  
    * L1-hinge loss  
    * L2-hinge loss  
* Confidence Weighted  
    * variance version  
* Multi-Class Multi-Label Confidence Weighted   
    * single contraint and diagonal matrix version  
* Adaptive Regularization of Weights  
    * dense matrix  
    * diagonal matrix  
* Softconfidence Weighted  
    * L1-hinge loss  
    * L2-hinge loss  
* Softconfidence Weighted  
    * L1-hinge loss, single contraint, and diagonal matrix  
    * L2-hinge loss, single contraint, and diagonal matrix  
* Logistic Regression (should not be used)  
    * solved with stochastic gradient  

## Dependency
* numpy
* scipy
* scikit-learn

## How to use
Refer to the main function

## Data Interface
* Learn inteface takes two arguments of X and y which are a 2d-numpy array or a scipy sparse martix and an array-like object respectively like scikit-learn fit interface.
* Predict interface takes one argument which is a 1d-numpy array or a 1-by-n scipy sparse matrix.

All you have to do is instantiate model, call learn method with samples and labels, an d predict for sample similar to scikit-learn interface

```python:main.py
# learn 
model = MCWVarDiag(eta=0.9, epochs=1)
model.learn(X, y) # X is samples and y is the correponding labels.

# predict
model.predict(x) # x is one samples.
```

## Data Format
This is a sample data format to be stored in your storage.
Any data format is acceptable unless you can feed data into learn/predict interface.

* Dense Labeled Samples
    * label,d1,d2,...,dn
    * example for binary

            1,0.1,0.2, ...,0.9  
            2,0.1,0.2, ...,0.9  
            ...  
            1,0.1,0.2, ...,0.9  
            2,0.1,0.2, ...,0.9  

    * example for multi-class  

            1,0.1,0.2, ...,0.9  
            2,0.1,0.2, ...,0.9  
            ...  
            3,0.1,0.2, ...,0.9  
            2,0.1,0.2, ...,0.9  

*  Sparse Labeled Samples

        label<space>feature-index:feature-val<space>feature-index:feature-val<space>...
        label<space>feature-index:feature-val<space>feature-index:feature-val<space>...
        ...
        label<space>feature-index:feature-val<space>feature-index:feature-val<space>...
    
## Note
* All labeled samples to be used for learning are stored in memory.
* All algorithm except for Multi-Class Multi-Label CW and SCW are for binary classification only.
* Bias, 1 is added to a sample, so you do not have to add 1 to dataset.

## References
* http://webee.technion.ac.il/people/koby/publications/arow_nips09.pdf
* http://www.aclweb.org/anthology/D/D09/D09-1052.pdf
* https://alliance.seas.upenn.edu/~nlp/publications/pdf/dredze2008f.pdf
* http://webee.technion.ac.il/people/koby/publications/paper_nips08_std.pdf
* http://icml.cc/2012/papers/86.pdf

## Future Work
* Labeled samples to be used for learning are not necessarily stored in memory.
* Evaluation compared to batch-learning (e.g., liblinear)
