import numpy as np
import scipy as sp
import logging as logger
import time
import pylab as pl
from collections import defaultdict
from sklearn.metrics import confusion_matrix

class PassiveAggressiveII(object):
    """
    Passive Aggressive-II algorithm: squared hinge loss PA.
    
    References: 
    - http://jmlr.org/papers/volume7/crammer06a/crammer06a.pdf
    
    This model is only applied to binary classification.
    """
    
    def __init__(self, fname, delimiter = " ", C = 1, n_scan = 10):
        """
        model initialization.
        """
        logger.basicConfig(level=logger.DEBUG)
        logger.info("init starts")

        self.n_scan = 10
        self.data = defaultdict()
        self.model = defaultdict()
        self.cache = defaultdict()
        self._load(fname, delimiter)
        self._init_model(C)
        
        logger.info("init finished")

    def _load(self, fname, delimiter = " "):
        """
        Load data set specified with filename.

        data format must be as follows (space-separated file as default),

        l_1 x_11 x_12 x_13  ... x_1m
        l_2 x_21 x_22  ... x_2m
        ...
        l_n x_n1 x_n2  ... x_nm

        l_i must be {1, -1} because of binary classifier.
        
        Arguments:
        - `fname`: file name.
        - `delimiter`: delimiter of a file.
        """
        logger.info("load data starts")
        
        # load data
        self.data["data"] = np.loadtxt(fname, delimiter = delimiter)
        self.data["n_sample"] = self.data["data"].shape[0] 
        self.data["f_dim"] = self.data["data"].shape[1] - 1

        # binalize
        self._binalize(self.data["data"])

        # normalize
        self.normalize(self.data["data"][:, 1:])

        logger.info("load data finished")
        
    def _binalize(self, data):
        """
        Binalize label of data.
        
        Arguments:
        - `data`: dataset.
        """
        logger.info("init starts")
        
        # binary check
        labels = data[:, 0]
        classes = np.unique(labels)
        if classes.size != 2:
            print "label must be a binary value."
            exit(1)

        # convert binary lables to {1, -1}
        for i in xrange(labels.size):
            if labels[i] == classes[0]: 
                labels[i] = 1
            else:
                labels[i] = -1

        # set classes
        self.data["classes"] = classes
        logger.info("init finished")

    def normalize(self, samples):
        """
        nomalize sample, such that sqrt(x^2) = 1
        
        Arguments:
        - `samples`: dataset without labels.
        """
        logger.info("normalize starts")
        for i in xrange(0, self.data["n_sample"]):
            samples[i, :] = self._normalize(samples[i, :])
            
        logger.info("normalize finished")

    def _normalize(self, sample):
        norm = np.sqrt(sample.dot(sample))
        sample = sample/norm
        return sample
        
    def _init_model(self, C):
        """
        Initialize model.
        """
        logger.info("init model starts")
        self.model["w"] = np.ndarray(self.data["f_dim"] + 1) # model paremter
        self.model["C"] = C                                      # aggressive parameter
        logger.info("init model finished")
        
    def _learn(self, ):
        """
        Learn internally.
        """

    def _update(self, label, sample, margin):
        """
        Update model parameter internally.
        update rule is as follows,
        w = w + y (1 - m)/(||x||_2^2 + C) * x
        Arguments:
        - `label`: label = {1, -1}
        - `sample`: sample, or feature vector
        """
        # add bias
        sample = self._add_bias(sample)

        norm = sample.dot(sample)
        w = self.model["w"] + label * (1 - margin)/(norm + self.model["C"]) * sample
        self.model["w"] = w
        
    def _predict_value(self, sample):
        """
        predict value of \w^T * x
        
        Arguments:
        - `sample`:
        """
        return self.model["w"].dot(self._add_bias(sample))

    def _add_bias(self, sample):
        return np.hstack((sample, 1))

    def learn(self, ):
        """
        Learn.
        """
        logger.info("learn starts")
        data = self.data["data"]
        
        # learn
        for i in xrange(0, self.n_scan):
            for i in xrange(0, self.data["n_sample"]):
                sample = data[i, 1:]
                label = data[i, 0]
                pred_val = self._predict_value(sample)
                margin = label * pred_val
                if  margin < 1:
                    self._update(label, sample, margin)

        logger.info("learn finished")

    def predict(self, sample):
        """
        predict {1, -1} base on \w^T * x
        
        Arguments:
        - `sample`:
        """
        pred_val = self._predict_value(sample)
        self.cache["pred_val"] = pred_val
        if pred_val >=0:
            return 1
        else:
            return -1
        
    def update(self, label, sample):
        """
        update model.
        Arguments:
        - `sample`: sample, or feature vector
        - `pred_val`: predicted value i.e., w^T * sample
        """
        margin = label * self.model["pred_val"]
        if  margin < 1:
            _update(label, sample, margin)

    @classmethod
    def examplify(cls, fname, delimiter = " ", C = 1 , n_scan = 3):
        """
        Example of how to use
        """
        
        # learn
        st = time.time()
        model = PassiveAggressiveII(fname, delimiter, C , n_scan)
        model.learn()
        et = time.time()
        print "learning time: %f[s]" % (et - st)

        # predict (after learning)
        data = np.loadtxt(fname, delimiter = " ")
        model._binalize(data)
        n_sample = data.shape[0]
        y_label = data[:, 0]
        y_pred = np.ndarray(n_sample)
        for i in xrange(0, n_sample):
            sample = data[i, 1:]
            y_pred[i] = model.predict(sample)

        # show result
        cm = confusion_matrix(y_label, y_pred)
        print cm
        print "accurary: %d [%%]" % (np.sum(cm.diagonal()) * 100.0/np.sum(cm))

if __name__ == '__main__':
    fname = "/home/kzk/datasets/uci_csv/liver.csv"
    #fname = "/home/kzk/datasets/uci_csv/ad.csv"
    print "dataset is", fname
    PassiveAggressiveII.examplify(fname, delimiter = " ", C = 1, n_scan = 100)
