import os

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(5, 4))

base_path = '/home/trevor/Documents/dev/ml/rl/rl-adaptive-sampling/rl_adaptive_sampling/lqr/data/variants2/'

datas = [
    ['envlqr_kf0_maxsamples300000_batch100_lr0.005_error0.0_diag0_sos0.0/1', 'PG batch 1'],
    ['envlqr_kf0_maxsamples300000_batch500_lr0.005_error0.0_diag0_sos0.0/1', 'PG batch 5'],
    ['envlqr_kf0_maxsamples300000_batch1000_lr0.005_error0.0_diag0_sos0.0/1', 'PG batch 10'],
    ['envlqr_kf0_maxsamples300000_batch5000_lr0.005_error0.0_diag0_sos0.0/1', 'PG batch 50'],

    ['envlqr_kf1_maxsamples300000_batch10000_lr0.005_error0.1_diag1_sos0.0/1', 'KF error 0.1'],
    ['envlqr_kf1_maxsamples300000_batch10000_lr0.005_error0.2_diag1_sos0.0/1', 'KF error 0.2'],
    ['envlqr_kf1_maxsamples300000_batch10000_lr0.005_error0.3_diag1_sos0.0/1', 'KF error 0.3'],
    ['envlqr_kf1_maxsamples300000_batch10000_lr0.005_error0.5_diag1_sos0.0/1', 'KF error 0.5'],
]

def plot_variant(data):
    end_path, name = data
    full_path = os.path.join(base_path, end_path)
    batches = np.load(os.path.join(full_path, 'log_batch_sizes.npy'))
    batch_ends = np.concatenate((np.array([0]), np.cumsum(batches)))
    eval_perf = np.load(os.path.join(full_path, 'log_eval_perf.npy'))

    print (eval_perf.shape)

    plt.plot(batch_ends[1:], eval_perf, label=name)

for d in datas:
    plot_variant(d)

colors = ['xkcd:coral', 'xkcd:tangerine', 'xkcd:scarlet'] #, 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']

plt.tight_layout()
plt.legend()

plt.xlabel('Rollouts')
plt.ylabel('Policy cost')

plt.show()
# plt.savefig(fname='lqr_point_mass.pdf', format='pdf')
