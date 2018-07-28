import os

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

# fig = plt.figure(figsize=(4, 3))

path = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/kf1_noisyobj0_fparabola_maxsamples5000_batch5000_lr0.01_error0.1_diag1_sos0.0/0/'
# path = '/home/trevor/Documents/dev/ml/rl/rl-adaptive-sampling/rl_adaptive_sampling/bandit/data/batch1000_lr0.1_error0.025_noisyobj0_fparabola_diag1_sos0.0/1/'

obs_noise = np.load(os.path.join(path, 'log_obs_noise_est.npy'))
batches = np.load(os.path.join(path, 'log_batch_sizes.npy'))
batch_ends = np.concatenate((np.array([0]), np.cumsum(batches)))

colors = ['xkcd:coral', 'xkcd:tangerine', 'xkcd:scarlet'] #, 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']


fig, ax1 = plt.subplots(figsize=(4,3))

ax1.bar(np.arange(len(batches)), batches, color='blue', label='batch size')
ax1.set_xlabel('Iteration')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('batch size')
# ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(np.squeeze(obs_noise[batch_ends[0:-1]]), color='orange', label='estimated variance')
ax2.set_ylabel('variance')
# ax2.tick_params('y', colors='orange')

plt.xlim((0, 25))
fig.legend()
fig.tight_layout()
plt.show()
# plt.savefig(fname='batch_var.pdf', format='pdf')

# f, (axes) = plt.subplots(1, 5, figsize=(12, 3))
# lims = [(-10, 10), (-10, 10), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
# print (axes)
# for batch in range(axes.size):
#     ax = axes.flatten()[batch]
#     # data = np.squeeze(grad_obs[batch_ends[batch]:batch_ends[batch+1]])
#     # ax.hist(data, bins=50, density=True)
#     # ax.set_xlim(lims[batch])
#     # ax.set_ylim((0.0, 0.5))
#     # if batch > 0:
#     #     ax.yaxis.set_visible(False)
#
#     ax.plot(np.squeeze(grad_est[batch_ends[batch]:batch_ends[batch+1]]), label='gradient estimate')
#     ax.plot(np.repeat(grad_trues[batch][0], batches[batch]), label='true gradient')
#     ax.plot(np.abs(np.squeeze(grad_est[batch_ends[batch]:batch_ends[batch+1]])-grad_trues[batch][0]), label='true gradient error')
#     ax.plot(np.squeeze(grad_errs[batch_ends[batch]:batch_ends[batch+1]]), label='expected gradient error')
#
# axes[2].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1,  borderaxespad=0, frameon=False)
# f.subplots_adjust(wspace=0, hspace=0)
# f.tight_layout()
# # plt.show()
#
# plt.savefig(fname='errors.pdf', format='pdf')
