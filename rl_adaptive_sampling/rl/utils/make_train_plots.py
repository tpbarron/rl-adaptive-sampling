import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from logger import DataLog

def make_train_plots(log_path = None,
                     keys = None,
                     save_loc = None):
    logger = DataLog()
    logger.read_log(log_path)
    log = logger.log
    print ("Log keys:", log.keys())
    # make plots for specified keys
    for key in keys:
        if key in log.keys():
            plt.figure(figsize=(10,6))
            plt.plot(log['total_samples'], log[key])
            plt.title(key)
            plt.show()
            # plt.savefig(save_loc+'/'+key+'.png', dpi=100)
            # plt.close()


if __name__ == "__main__":
    import sys
    keys = ["eval_perf"]
    make_train_plots(None, sys.argv[1], keys)
