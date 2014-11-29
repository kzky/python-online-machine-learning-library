#!/usr/bin/env python

import pylab as pl
import json
import numpy as np

# vars
results_filepath = "./evaluate_small_data_results.json"
#image_dirpath = "/home/k_yoshiyama/images/uci_small_dataset"
image_dirpath = "/home/kzk/images/uci_small_dataset"

# save as image
results = json.load(open(results_filepath))
data_names = results.keys()
data_names.sort()
for data_name in data_names:
    results_per_data = results[data_name]
    model_names = results_per_data.keys()
    model_names.sort()

    # acc
    fig = pl.figure()
    for model_name in model_names:
        pl.plot(results[data_name][model_name]["acc"], label=model_name)
        pass
    pl.legend(loc="lower right")
    pl.xlabel("epochs")
    pl.ylabel("%")
    pl.title("Accuracy for %s" % data_name)
    pl.savefig("%s/acc_%s.png" % (image_dirpath, data_name), dpi=200)
    pl.clf()

    # elapsed time
    for model_name in model_names:
        if not model_name == "LinearSVC":
            pl.plot(np.cumsum(results[data_name][model_name]["elapsed"]), label=model_name)
        else:
            pl.plot(results[data_name][model_name]["elapsed"], label=model_name)
        pass
     
    pl.legend(loc="lower right")
    pl.xlabel("epochs")
    pl.ylabel("sec")
    pl.title("Cummulative Elapsed Time for %s" % data_name)
    pl.savefig("%s/cum_elapsed_%s.png" % (image_dirpath, data_name), dpi=200)
    pl.clf()
    
    pass
pass
