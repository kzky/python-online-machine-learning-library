import numpy as np
import scipy as sp
import logging as logger
import time
import pylab as pl
from collections import defaultdict
from sklearn.metrics import confusion_matrix

###################
## should be refactored   ##
###################

class MultiClassLogisticRegression(object):
    """
    Mulit Class Logistic Regression solved with stochastic gradient descent.
    Applicable only to linear model.
    
    Should be refactored.
    """
    def __init__(self, epsilon = 0.05, n_scan = 500):
        """
        Initializer.
        Arguments:
        - `epsilon`: step size for stochastic gradient.
        - `n_scan`: n_scan
        """
        logger.basicConfig(level=logger.DEBUG)
        logger.info("init starts")
        self.data = None
        self.model = defaultdict()
        self.model["epsilon"] = epsilon
        self.model["n_scan"] = n_scan
        self.epsilon = epsilon
        logger.info("init finished")

    def _load(self, fname, delimiter=" "):
        """ 
        Load data with file name. 
        data format must be as follows (space-separated file as default),

        l_1 x_11 x_12 x_13  ... x_1m
        l_2 x_21 x_22  ... x_2m
        ...
        l_n x_n1 x_n2  ... x_nm

        Arguments:
        - `fname`: File name to be loaded.
        """
        logger.info("load data starts")
        
        # load data
        self.data = np.loadtxt(fname, delimiter = delimiter)
        
        logger.info("load data finished")

    def _init_model(self):
        """
        Initialize model.
        """
        logger.info("init model starts")

        # class label/num of classes
        classes = np.unique(self.data[:, 0])
        self.model["classes"] = classes
        self.model["n_classes"] = classes.size 

        # theta's dimension/ number of samples
        self.model["n_samples"] = self.data.shape[0] 
        self.model["f_dims"] = self.data.shape[1] - 1

        # map[label, vector]/probability
        self.model["theta"] = defaultdict()
        self.model["probs"] = defaultdict()
        for i in self.model["classes"]:
            self.model["theta"][i] = np.random.rand(self.model["f_dims"]) - 0.5
            self.model["probs"][i] = 0.0

        # model information
        logger.info("####### Model Info ###########")
        logger.info("feature dimensions: %d" % self.model["f_dims"])
        logger.info("number of samples: %d" % self.model["n_samples"])
        logger.info("number of classes: %d" % self.model["n_classes"])
        logger.info("classes: %s" % self.model["classes"])
        logger.info("thetas: %s" % self.model["theta"])
        logger.info("probability: %s" % self.model["probs"])
        logger.info("##########################")

        logger.info("init model finished")

    def learn(self, fname):
        """
        Learn through dataset specified by fname with the number of scan.
        
        Arguments:
        - `fname`: dataset.
        """
        # load dataset
        self._load(fname)

        # init model info
        self._init_model()

        # learn
        logger.info("learning starts")
        for c in xrange(0, self.model["n_scan"]): 
            for i in xrange(0, self.model["n_samples"]):
                self._learn(self.data[i, :])

        logger.info("learning finished")

    def _learn(self, l_sample):
        """
        learn method internally.
        """
        # predict
        probs = self._predict_probs(l_sample[1:])
        
        # update
        self._update(l_sample, probs)
        
    def _update(self, l_sample, probs):
        """
        Update model internally.

        Update rule is as follows.
        theta_c = theta_c + epsilon * delta(J(theta)) for all c.
        delta(J) = - exp(theta_c^T * x)/sum_c {exp(theta_c^T * x} * x + x ( if c = y )
        delta(J) = - exp(theta_c^T * x)/sum_c {exp(theta_c^T * x} * x ( if c != y )
        """
        # self.epsilon = self.epsilon/2 # huristics update for step size
        for cls in self.model["classes"]:
            self.model["theta"][cls] = self.model["theta"][cls] - self.epsilon * probs[cls] *l_sample[1:] 
            
        self.model["theta"][l_sample[0]] = self.model["theta"][l_sample[0]] + self.epsilon * l_sample[1:]
                
    def _predict_probs(self, sample):
        """
        predict samples with probability internally.
        Arguments:
        - `sample`: 1-dimensional ndarray.
        """
        # inner prods
        inner_prods = defaultdict()
        for cls in self.model["classes"]:
            inner_prods[cls] = self.model["theta"][cls].dot(sample)
        max_ = max(inner_prods.values())

        # probs
        probs = defaultdict()
        sum_ = 0
        for cls in self.model["classes"]:
            probs[cls] = np.exp(inner_prods[cls] - max_)
            sum_ += probs[cls]
    
        # normalize probs
        for cls in probs.keys():
            probs[cls] = probs[cls]/sum_

        return probs

    def predict_probs(self, fname):
        """
        predict probability with leaned model

        Arguments:
        - `fname`:
        """
        logger.info("predict starts")
        data = np.loadtxt(fname)
        n_samples = data.shape[0]
        outputs = defaultdict()
        for i in xrange(0, n_samples):
            outputs[i] = self._predict_probs(data[i, 1:])

        logger.info("predict finished")

        return outputs

    def predict(self, fname):
        """
        predict label with leaned model
        Arguments:
        - `fname`:
        """
        outputs_ = self.predict_probs(fname)
        outputs = np.ndarray(len(outputs_))

        for i in xrange(0, len(outputs)):
            outputs[i] = max(outputs_[i], key=outputs_[i].get)
                        
        return outputs

    def predict_probs_then_update(self, sample):

        # predict
        probs = self._predict_probs(sample)
        
        # update
        self._update(sample, probs)
        
        return probs
    
    @classmethod
    def examplify(cls, fname):
        """
        example of how to use
        """
        logger.info("examplify starts")
        
        # model
        model = MultiClassLogisticRegression(epsilon=0.01, n_scan = 100)
        
        # learn
        st = time.time()
        model.learn(fname)
        et = time.time()
        print "learning time: %d [s]" % ((et - st)/1000)

        # predict
        y_pred = model.predict(fname)
        
        # confusion matrix
        y_label = np.loadtxt(fname, delimiter=" ")[:, 0]

        cm = confusion_matrix(y_label, y_pred)
        #pl.matshow(cm)
        #pl.title('Confusion matrix')
        #pl.colorbar()
        #pl.ylabel('True label')
        #pl.xlabel('Predicted label')
        #pl.show()

        print cm
        print "accurary: %d [%%]" % (np.sum(cm.diagonal()) * 100.0/np.sum(cm))
        logger.info("examplify finished")

if __name__ == '__main__':
    # TODO
    # stopping criterion
    # epsilon/n_scan treatment
    #fname = "/home/kzk/datasets/uci_csv/glass.csv"
    fname = "/home/kzk/datasets/uci_csv/liver.csv"
    MultiClassLogisticRegression.examplify(fname)
        
    

