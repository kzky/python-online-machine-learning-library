#!/usr/bin/env python

import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix
from collections import defaultdict


class ConfidenceWeightedModel(object):
    """
    """
    
    def __init__(self, epochs=10):
        """
        """
        self.epochs = epochs
        self.data = defaultdict()
        self.model = defaultdict()
        self.cache = defaultdict()
        self.data["one"] = csr_matrix(([1], ([0], [0])))
        
        pass
    
    def _add_bias_for_dense_sample(self, sample):
        return np.hstack((sample, 1))
    
    def _add_bias_for_sparse_sample(self, sample):
        """
        
        Arguments:
        - `sample`:
        """
        x = sparse.hstack([sample, self.data["one"]])
        #return x.tocsr()
        return x

    def inverse_1d_sparse_matrix(self, X):
        """
        Disruptive method.
        
        Arguments:
        - `X`:
        """
        X.data = 1 / X.data
        return X

