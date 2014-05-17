import numpy as np
import scipy as sp
import logging as logger
import time
import pylab as pl
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from scipy.stats import norm

class SCWII(object):
    """
    Full matrix version of Soft Confidence-Weighted algorithm with L2-hinge loss.
    
    References: 
    - http://icml.cc/2012/papers/86.pdf
    - https://alliance.seas.upenn.edu/~nlp/publications/pdf/dredze2008f.pdf
    - http://webee.technion.ac.il/people/koby/publications/paper_nips08_std.pdf

    Note:
    - This model is only applied to binary classification.
    """

    def __init__(self, fname, delimiter = " ", eta = 0.9, C = 1, n_scan = 10):
        """
        model initialization.
        """
        logger.basicConfig(level=logger.DEBUG)
        logger.info("init starts")

        self.n_scan = n_scan
        self.data = defaultdict()
        self.model = defaultdict()
        self.cache = defaultdict()
        self._load(fname, delimiter)
        self._init_model(eta, C)
        
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
        st = time.time()
        self.data["data"] = np.loadtxt(fname, delimiter = delimiter)
        et = time.time()
        logger.info("loading data time: %f[s]", (et - st))
        self.data["n_sample"] = self.data["data"].shape[0] 
        self.data["f_dim"] = self.data["data"].shape[1] - 1

        # binalize
        self._binalize(self.data["data"])
        
        # normlize
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

    def _init_model(self, eta, C):
        """
        Initialize model.
        """
        logger.info("init model starts")
        self.model["mu"] = np.zeros(self.data["f_dim"] + 1) # model parameter mean
        self.model["C"] = C
        self.model["S"] = np.identity(self.data["f_dim"] + 1)#model parameter covariance
        self.model["eta"] = eta                                            # confidence parameter
        self.model["phi"] = norm.ppf(norm.cdf(eta))      # inverse of cdf(eta)
        self.model["phi_2"] = self.model["phi"] ** 2
        self.model["phi_4"] = self.model["phi_2"] ** 2
        self.model["psi"] = 1 + self.model["phi_2"] / 2
        self.model["zeta"] = 1 + self.model["phi_2"]
        logger.info("init model finished")
        
    def _learn(self, ):
        """
        Learn internally.
        """

    def _update(self, label, sample, m, v):
        """
        Update model parameter internally.

        Arguments:
        - `label`: label = {1, -1}
        - `sample`: sample, or feature vector
        """

        # add bias
        sample = self._add_bias(sample)

        # alpha
        C = self.model["C"]
        n = v + 1/(2*C)
        phi = self.model["phi"]
        phi_2 = self.model["phi_2"]
        gamma = phi * np.sqrt((phi*m*v)**2 + 4*n*v*(n + v*phi_2))
        alpha = -((2*m*n + phi_2*m*v) + gamma)/(2*(n**2 + n*v*phi_2))
        alpha = max(0, alpha)
        
        # mu
        S = self.model["S"]
        mu = self.model["mu"] + alpha * label * S.dot(sample)
        self.model["mu"] = mu
        
        # beta
        alpha_v_phi = alpha*v*phi
        u = ((-alpha_v_phi + np.sqrt(alpha_v_phi**2 + 4*v)) ** 2) / 4
        beta = alpha * phi / (np.sqrt(u) +  alpha_v_phi)

        # S
        S = S - beta * S.dot(np.outer(sample, sample)).dot(S)
        self.model["S"] = S
        
    def _predict_value(self, sample):
        """
        predict value of \mu^T * x
        
        Arguments:
        - `sample`:
        """
        return self.model["mu"].dot(self._add_bias(sample))
    
    def _add_bias(self, sample):
        return np.hstack((sample, 1))

    def learn(self, ):
        """
        Learn.
        """
        logger.info("learn starts")
        data = self.data["data"]
        
        # learn
        st = time.time()
        for i in xrange(0, self.n_scan):
            print "iter:", i
            for i in xrange(0, self.data["n_sample"]):
                sample = data[i, 1:]
                label = data[i, 0]
                pred_val = self._predict_value(sample)
                m = label * pred_val
                biased_sample = self._add_bias(sample)
                v = biased_sample.dot(self.model["S"]).dot(biased_sample)
                if  m < self.model["phi"] * np.sqrt(v):
                    self._update(label, sample, m, v)

        logger.info("learn finished")
        et = time.time()
        logger.info("learning time: %f[s]" % (et - st))

    def predict(self, sample):
        """
        predict {1, -1} base on \mu^T * x
        
        Arguments:
        - `sample`:
        """
        pred_val = self._predict_value(sample)
        self.cache["pred_val"] = pred_val
        if pred_val >=0:
            return 1
        else:
            return -1

    ## TODO
    def update(self, label, sample):
        """
        update model.
        Arguments:
        - `sample`: sample, or feature vector
        - `pred_val`: predicted value i.e., mu^T * sample
        """

        margin = label * self.model["pred_val"]
        if  margin < 1:
            _update(label, sample, margin)

    @classmethod
    def examplify(cls, fname, delimiter = " ", eta = 0.1, C = 1, n_scan = 1):
        """
        Example of how to use
        """
        
        # learn
        model = SCWII(fname = fname, delimiter = delimiter, eta = eta, C = C, n_scan = n_scan)
        model.learn()

        # predict (after learning)
        data = np.loadtxt(fname, delimiter = " ")
        model._binalize(data)
        model.normalize(data[:, 1:])
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
    #fname = "/home/kzk/datasets/uci_csv/liver.csv"
    #fname = "/home/kzk/datasets/uci_csv/ad.csv"
    #fname = "/home/kzk/datasets/uci_csv/adult.csv"
    fname = "/home/kzk/datasets/uci_csv/iris2.csv"
    print "dataset is", fname
    SCWII.examplify(fname, delimiter = " ", eta = 0.9, C = 1, n_scan = 10)
