#!/usr/bin/env python

import numpy as np
import logging as logger
import time
import json
import copy


from sklearn.metrics import confusion_matrix
from multiclass_confidence_weighted_var_diag import MCWVarDiag
from multiclass_soft_confidence_weighted_1_diag import MSCWIDiag
from multiclass_soft_confidence_weighted_2_diag import MSCWIIDiag
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC

# file path
filepath = "./evaluate_sparse_data_results.json"

# data cnofiguration
data_config = {
    "rcv1": {
        "train": "/home/k_yoshiyama/datasets/rcv1/rcv1_train.multiclass.dat",
        "test": "/home/k_yoshiyama/datasets//rcv1/rcv1_test.multiclass.dat",
    },

    "sector": {
        "train": "/home/k_yoshiyama/datasets/sector/sector.scale.dat",
        "test": "/home/k_yoshiyama/datasets/sector/sector.t.scale.dat",
    },
}

# results
results = {
    "rcv1": {
    },
    "sector": {
    },
}

# model config
models = [MCWVarDiag, MSCWIDiag, MSCWIIDiag]
model_class_name_map = {
    MCWVarDiag: "MCWVarDiag",
    MSCWIDiag: "MSCWIDiag",
    MSCWIIDiag: "MSCWIIDiag",
}

# results
result_per_data = {
    "MCWVarDiag": {
        "acc": [],  # per epoch
        "elapsed": [],  # per epoch
    },

    "MSCWIDiag": {
        "acc": [],
        "elapsed": [],
    },

    "MSCWIIDiag": {
        "acc": [],
        "elapsed": [],
    },

    "LinearSVC": {
        "acc": [],
        "elapsed": [],
    },
}

# results for each data
for data in results:
    results[data] = copy.deepcopy(result_per_data)
    pass

# run experiment
epochs = xrange(1, 6)
for data in data_config:
    print "data %s is processing..." % data
    
    # train/test
    (X_train, y_train) = load_svmlight_file(data_config[data]["train"])
    (X_test, y_test) = load_svmlight_file(data_config[data]["test"])

    # evaluate
    for model in models:  # foreach __main__.class
        # init
        print "model is %s" % str(model)
        model_ = model(epochs=1)
        print "model is %s." % model_class_name_map[model]

        # epoch
        for epoch in epochs:
            print "the number of epochs is %d" % epoch
            # warm start
            if not epoch == 1:
                mu = model_.model["mu"]
                S = model_.model["S"]
                model_.init_params(mu, S)
                pass

            # learn
            st = time.time()
            model_.epochs = 1
            model_.learn(X_train, y_train)
            et = time.time()

            # elapsed time
            results[data][model_class_name_map[model]]["elapsed"].append(et - st)
            
            # predict
            y_pred = []
            for x in X_test:
                y_pred.append(model_.predict(x))
                pass
            cm = confusion_matrix(y_test, y_pred)

            # accuracy
            results[data][model_class_name_map[model]]["acc"].append(np.sum(cm.diagonal()) * 100.0 / np.sum(cm))
        
            pass
        pass

    # Linear SVC
    print "model is LinearSVC."
    model_ = LinearSVC()
    st = time.time()
    model_.fit(X_train, y_train)
    et = time.time()
    y_pred = model_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    for epoch in epochs:  # add the same results to all epochs
        results[data]["LinearSVC"]["acc"].append(np.sum(cm.diagonal()) * 100.0 / np.sum(cm))
        results[data]["LinearSVC"]["elapsed"].append(et - st)
        pass
    pass

with open(filepath, "w") as fpout:
    json.dump(results, fpout)
    pass

