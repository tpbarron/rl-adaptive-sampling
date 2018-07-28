import os

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

# fig = plt.figure(figsize=(4, 3))

path = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/kf1_noisyobj0_fparabola_maxsamples5000_batch5000_lr0.01_error0.1_diag1_sos0.0/0/'
# path = '/home/trevor/Documents/dev/ml/rl/rl-adaptive-sampling/rl_adaptive_sampling/bandit/data/batch1000_lr0.1_error0.025_noisyobj0_fparabola_diag1_sos0.0/1/'

grad_trues = np.load(os.path.join(path, 'log_grad_true.npy'))
grad_est = np.load(os.path.join(path, 'log_grad_est.npy'))
batches = np.load(os.path.join(path, 'log_batch_sizes.npy'))
batch_ends = np.concatenate((np.array([0]), np.cumsum(batches)))
grad_errs = np.load(os.path.join(path, 'log_abs_error_est.npy'))

colors = ['xkcd:coral', 'xkcd:tangerine', 'xkcd:scarlet'] #, 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']

f, (axes) = plt.subplots(1, 6, sharey=True, figsize=(10, 1.7))
lims = [(-10, 10), (-10, 10), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
# print (axes)
batchids = [0, 25, 50, 100, 250, 400] #, 1000, 1500, 2000, 2400]
for i in range(axes.size):
    print (i)
    batch = batchids[i]
    ax = axes.flatten()[i]
    # data = np.squeeze(grad_obs[batch_ends[batch]:batch_ends[batch+1]])
    # ax.hist(data, bins=50, density=True)
    # ax.set_xlim(lims[batch])
    # ax.set_ylim((0.0, 0.5))
    # if batch > 0:
    #     ax.yaxis.set_visible(False)

    # if i > 0:
        # ax.set_xticks([0, 1])
    ax.set_xlabel("Iteration " + str(batch))
    ax.plot(np.squeeze(grad_est[batch_ends[batch]:batch_ends[batch+1]]), label='gradient estimate')
    ax.plot(np.repeat(grad_trues[batch][0], batches[batch]), label='true gradient')

    ax.plot(np.abs(np.squeeze(grad_est[batch_ends[batch]:batch_ends[batch+1]])-grad_trues[batch][0]), label='true gradient error', linestyle='dashed')
    ax.plot(np.squeeze(grad_errs[batch_ends[batch]:batch_ends[batch+1]]), label='expected gradient error', linestyle='dashed')

plt.legend(loc="upper left", bbox_to_anchor=(1,1))

# axes[2].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1,  borderaxespad=0, frameon=False)
f.subplots_adjust(wspace=0, hspace=0)
f.tight_layout()
plt.show()

# plt.savefig(fname='errors.pdf', format='pdf')
