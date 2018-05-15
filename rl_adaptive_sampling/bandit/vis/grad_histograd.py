import os

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

# fig = plt.figure(figsize=(4, 3))

path = '/home/trevor/Documents/dev/ml/rl/rl-adaptive-sampling/rl_adaptive_sampling/bandit/data/batch500_lr0.1_error0.025_noisyobj0_fparabola_diag1_sos0.0/1/'
grad_obs = np.load(os.path.join(path, 'log_grad_obs.npy'))
batches = np.load(os.path.join(path, 'log_batch_sizes.npy'))
batch_ends = np.concatenate((np.array([0]), np.cumsum(batches)))


colors = ['xkcd:coral', 'xkcd:tangerine', 'xkcd:scarlet'] #, 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']


f, (axes) = plt.subplots(1, 8, sharey=True, figsize=(8, 1.25))
# lims = [(-10, 10), (-10, 10), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
for batch in range(len(axes)):
    ax = axes[batch]
    data = np.squeeze(grad_obs[batch_ends[batch]:batch_ends[batch+1]])
    (w,p) = ss.shapiro(data)
    # print (w,p)
    # input("")
    ax.hist(data, bins=50, density=True)
    ax.annotate(str(np.round(w, decimals=2)), xy=(1.0, 0.25))
    # ax.text(0.9, 0.25, str(np.round(w, decimals=2)))
    # ax.set_xlim(lims[batch])
    ax.set_ylim((0.0, 0.5))
    # if batch > 0:
        # ax.yaxis.set_visible(False)
f.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()
# plt.show()
plt.savefig(fname='gradhist.pdf', format='pdf')
