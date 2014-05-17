import numpy as np
import scipy as sp
import logging as logger
import time
import pylab as pl
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from scipy.stats import norm

class MSCWIIDiag(object):
    """
    Diagonal elements of matrix version of Soft Confidence-Weighted II algorithm; 
    non-diagonal elements in covariance matrix are ignored.
    
    References:
    - http://www.aclweb.org/anthology/D/D09/D09-1052.pdf
    - http://icml.cc/2012/papers/86.pdf

    Feature function F(x, y) is chosen as cartesian product of x and y.
    x is feature vector and y is 1-of-K vector.

    This model is applied to multiclass-multilabel classification, solved with
    single constraint update in http://www.aclweb.org/anthology/D/D09/D09-1052.pdf.
    """

    def __init__(self, fname, delimiter = " ", C = 1, eta = 0.1, n_scan = 3):
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
        self._init_model(C, eta)
        
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
        self.data["classes"] = np.unique(self.data["data"][:, 0])

        # normlize
        self.normalize(self.data["data"][:, 1:])
        
        logger.info("load data finished")
        
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

    def _init_model(self, C, eta):
        """
        Initialize model.
        """
        logger.info("init model starts")
        self.model["mu"] = defaultdict() # model parameter mean
        self.model["S"] = defaultdict()    #model parameter covariance
        self.model["C"] = C                       # PA parameter
        for k in self.data["classes"]:
            self.model["mu"][k] = np.zeros(self.data["f_dim"] + 1) 
            self.model["S"][k] = np.ones(self.data["f_dim"] + 1)  # only for diagonal
        self.model["eta"] = eta                                            # confidence parameter
        self.model["phi"] = norm.ppf(norm.cdf(eta))      # inverse of cdf(eta)
        self.model["phi_2"] = np.power(self.model["phi"], 2)
        self.model["psi"] = 1 + self.model["phi_2"]/2
        self.model["zeta"] = 1 + self.model["phi_2"]
        logger.info("init model finished")
        
    def _learn(self, ):
        """
        Learn internally.
        """

    def _update(self, sample, y, r):
        """
        Update model parameter internally.
        update rule is as follows,
        mu = mu + alpha * y * Sx
        S = (S^{-1} + 2 * alpha * phi * diag(g_{y, r}^2))^{-1} 
        g_{y, r} = F(x, y) - F(x, r)

        Note: diagonal elements are only considered.
        
        Arguments:
        - `sample`: sample, or feature vector
        - `y`: true label
        - `r`: predicted label (!=y) with high rank value
        """

        # components
        phi = self.model["phi"]
        phi_2 = self.model["phi_2"]
        psi = self.model["psi"]
        zeta = self.model["zeta"]

        sample = self._add_bias(sample)
        g_y = sample
        g_r = -sample
        m = self.model["mu"][y].dot(g_y) + self.model["mu"][r].dot(g_r)
        v = (g_y * self.model["S"][y]).dot(g_y) + (g_r * self.model["S"][r]).dot(g_r)
        n = v + 1/(2*self.model["C"])
        n_2 = np.power(n, 2)
        m_2 = np.power(m, 2)
        v_2 = np.power(v, 2)
        gamma = phi * (np.sqrt(phi_2*m_2*v_2 + 4*n*v*(n+v*phi_2)))
        
        # alpha
        alpha = max(0, (-(2*m*n + phi_2*m*v) + gamma)/(2*(n_2 + n*v*phi_2)))
        
        # mu
        mu_y = self.model["mu"][y] + alpha*self.model["S"][y]*g_y
        mu_r = self.model["mu"][r] + alpha*self.model["S"][r]*g_r
        self.model["mu"][y] = mu_y
        self.model["mu"][r] = mu_r

        # beta
        alpha_2 = alpha * alpha
        v_2 = v * v
        u = -alpha * v * phi + np.sqrt(alpha_2*v_2*phi_2 + 4*v)
        u = u * u/4
        beta = (alpha * phi)/(np.sqrt(u) + v*alpha*phi)
        
        # S (only diagonal)
        S_y = self.model["S"][y] - beta * self.model["S"][y]*self.model["S"][y]*g_y*g_y
        S_r = self.model["S"][r] - beta * self.model["S"][r]*self.model["S"][r]*g_r*g_r
        self.model["S"][y] = S_y
        self.model["S"][r] = S_r
        
    def _predict_values(self, sample):
        """
        predict value of \mu^T * x
        
        Arguments:
        - `sample`:
        """

        values = defaultdict()
        sample = self._add_bias(sample)
        for k in self.data["classes"]:
            values[k] = self.model["mu"][k].dot(sample)

        # return as list of tuple (class, ranking) in descending order
        return [(k, v) for k, v in sorted(values.items(), key=lambda x:x[1], reverse=True)]

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
                pred_vals = self._predict_values(sample)
                high_rank_class = pred_vals[0][0]
                if high_rank_class != label:
                    self._update(sample, label, high_rank_class) # highest rank class

        logger.info("learn finished")
        et = time.time()
        logger.info("learning time: %f[s]" % (et - st))

    def predict(self, sample):
        """
        predict class base on argmax_{z} w^T F(x, z)
        
        Arguments:
        - `sample`:
        """
        pred_vals = self._predict_values(sample)
        self.cache["pred_vals"] = pred_vals
        return pred_vals[0][0]
        
    ## TODO
    def update(self, label, sample):
        """
        update model.
        Arguments:
        - `label`: label
        - `sample`: sample, or feature vector
        """

    @classmethod
    def examplify(cls, fname, delimiter = " ", C = 1, eta = 0.1, n_scan = 1):
        """
        Example of how to use
        """
        
        # learn
        model = MSCWIIDiag(fname = fname, delimiter = delimiter, C = C, eta = eta, n_scan = n_scan)
        model.learn()

        # predict (after learning)
        data = np.loadtxt(fname, delimiter = " ")
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
    #fname = "/home/kzk/datasets/uci_csv/iris.csv"
    #fname = "/home/kzk/datasets/uci_csv/glass.csv"
    fname = "/home/kzk/datasets/uci_csv/car.csv"
    fname = "/home/kzk/datasets/uci_csv/usps.csv"
    fname = "/home/kzk/datasets/uci_csv/yeast.csv"
    print "dataset is", fname

    # many iteration is necesary
    MSCWIIDiag.examplify(fname, delimiter = " ", C = 0.01, eta = 0.9, n_scan = 100)

