import numpy as np
import logging as logger
import time
from collections import defaultdict
from sklearn.metrics import confusion_matrix


class AROW2(object):
    """
    Adaptive Regularization of Weight Vector algorithm with squared hinge loss.

    References:
    - http://webee.technion.ac.il/people/koby/publications/arow_nips09.pdf
    
    This model is only applied to binary classification.
    """

    def __init__(self, C=1, epochs=10):
        """
        model initialization.
        """
        logger.basicConfig(level=logger.DEBUG)
        logger.info("init starts")

        self.epochs = epochs
        self.model = defaultdict()
        self.cache = defaultdict()
        self._init_model(C)
        
        logger.info("init finished")
        pass
    
    def _init_model(self, C):
        """
        Initialize model.
        """
        logger.info("init model starts")
        self.model["C"] = C  # aggressive parameter
        logger.info("init model finished")
        pass
    
    def _learn(self, ):
        """
        Learn internally.
        """
        pass

    def _update(self, label, sample, margin):
        """
        Update model parameter internally.

        Arguments:
        - `label`: label = {1, -1}
        - `sample`: sample, or feature vector
        """

        # add bias
        sample = self._add_bias(sample)

        # beta
        beta = sample.dot(self.model["S"]).dot(sample) + self.model["C"]
        
        # mu
        Sx = self.model["S"].dot(sample)
        mu = self.model["mu"] + label * (1 - margin) * Sx / beta
        self.model["mu"] = mu

        # S
        outer_dot_sample = np.outer(sample, sample)
        SoS = self.model["S"].dot(outer_dot_sample).dot(self.model["S"])
        S = self.model["S"] - SoS / beta
        self.model["S"] = S
        pass
    
    def _predict_value(self, sample):
        """
        predict value of \mu^T * x
        
        Arguments:
        - `sample`:
        """
        
        return self.model["mu"].dot(self._add_bias(sample))

    def _add_bias(self, sample):
        return np.hstack((sample, 1))

    def learn(self, X, y):
        """
        Learn.
        """
        logger.info("learn starts")
        self.model["n_samples"] = X.shape[0]
        self.model["f_dims"] = X.shape[1]

        # model parameter mean
        self.model["mu"] = np.zeros(self.model["f_dims"] + 1)

        # model parameter covariance
        self.model["S"] = np.identity(self.model["f_dims"] + 1)
        
        # learn
        st = time.time()
        for i in xrange(0, self.epochs):
            logger.info("iter: %d" % i)
            for i in xrange(0, self.model["n_samples"]):
                sample = X[i, :]
                label = y[i]
                pred_val = self._predict_value(sample)
                margin = label * pred_val
                if margin < 1:
                    self._update(label, sample, margin)

        logger.info("learn finished")
        et = time.time()
        logger.info("learning time: %f[s]" % (et - st))
        pass
    
    def predict(self, sample):
        """
        predict {1, -1} base on \mu^T * x
        
        Arguments:
        - `sample`:
        """
        pred_val = self._predict_value(sample)
        self.cache["pred_val"] = pred_val
        if pred_val >= 0:
            return 1
        else:
            return -1
        pass
    
    def update(self, label, sample):
        """
        update model.
        Arguments:
        - `sample`: sample, or feature vector
        - `pred_val`: predicted value i.e., mu^T * sample
        """
        margin = label * self.model["pred_val"]
        if margin < 1:
            self._update(label, sample, margin)
        pass
    pass

def main():
    """
    Example of how to use
    """
        
    # data load
    fname = "/home/kzk/datasets/uci_csv/liver.csv"
    #fname = "/home/kzk/datasets/uci_csv/ad.csv"
    #fname = "/home/kzk/datasets/uci_origin/adult.csv"
    #fname = "/home/kzk/datasets/uci_csv/adult.csv"
    print "dataset is", fname
    data = np.loadtxt(fname, delimiter=" ")
    X = data[:, 1:]
    y = data[:, 0]

    # learn
    model = AROW2(C=1, epochs=3)
    model.learn(X, y)

    # predict
    y_pred = np.ndarray(X.shape[0])
    for i in xrange(0, X.shape[0]):
        sample = data[i, 1:]
        y_pred[i] = model.predict(sample)

    # show result
    cm = confusion_matrix(y, y_pred)
    print cm
    print "accurary: %d [%%]" % (np.sum(cm.diagonal()) * 100.0 / np.sum(cm))

if __name__ == '__main__':
    main()
