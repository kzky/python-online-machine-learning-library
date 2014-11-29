#!/usr/bin/env python

import numpy as np
import time
import json
import copy

from sklearn.metrics import confusion_matrix
from multiclass_confidence_weighted_var_diag import MCWVarDiag
from multiclass_soft_confidence_weighted_1_diag import MSCWIDiag
from multiclass_soft_confidence_weighted_2_diag import MSCWIIDiag
from sklearn.svm import LinearSVC

# file path
filepath = "./evaluate_small_data_results.json"

# data cnofiguration
data_config = {
    "abalone": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/abalone.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/abalone.csv",
    },
    "transfusion": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/transfusion.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/transfusion.csv",
    },
    "gisette": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/gisette.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/gisette.csv",
    },
    "iris": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/iris.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/iris.csv",
    },
    "glass": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/glass.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/glass.csv",
    },
    "breast_cancer": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/breast_cancer.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/breast_cancer.csv",
    },
    "car": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/car.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/car.csv",
    },
    "creadit": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/credit.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/credit.csv",
    },
    "usps": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/usps.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/usps.csv",
    },
    "liver": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/liver.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/liver.csv",
    },
    "haberman": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/haberman.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/haberman.csv",
    },
    "pima": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/pima.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/pima.csv",
    },
    "ionosphere": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/ionosphere.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/ionosphere.csv",
    },
    "isolet": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/isolet.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/isolet.csv",
    },
    "magicGamaTelescope": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/magicGamaTelescope.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/magicGamaTelescope.csv",
    },
    "mammographic": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/mammographic.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/mammographic.csv",
    },
    "yeast": {
        "train": "/home/k_yoshiyama/datasets/uci_csv_train/yeast.csv",
        "test": "/home/k_yoshiyama/datasets/uci_csv_test/yeast.csv",
    },
}

# results
results = {
    "abalone": {
    },
    "transfusion": {
    },
    "gisette": {
    },
    "iris": {
    },
    "glass": {
    },
    "breast_cancer": {
    },
    "car": {
    },
    "creadit": {
    },
    "usps": {
    },
    "liver": {
    },
    "haberman": {
    },
    "pima": {
    },
    "ionosphere": {
    },
    "isolet": {
    },
    "magicGamaTelescope": {
    },
    "mammographic": {
    },
    "yeast": {
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
epochs = xrange(1, 51)
for data in data_config:
    print "data %s is processing..." % data
    
    # train/test
    data_train = np.loadtxt(data_config[data]["train"], delimiter=" ")
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]

    data_test = np.loadtxt(data_config[data]["test"], delimiter=" ")
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
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
    acc = np.sum(cm.diagonal()) * 100.0 / np.sum(cm)
    elapsed_time = et - st
    for epoch in epochs:  # add the same results to all epochs
        results[data]["LinearSVC"]["acc"].append(acc)
        results[data]["LinearSVC"]["elapsed"].append(elapsed_time)
        pass

    
with open(filepath, "w") as fpout:
    json.dump(results, fpout)
    pass

