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

path = '/run/media/trevor/01CA-028A/nips_kalman/lqr/5_17_18r0.1/envlqr_kf1_maxsamples500000_batch10000_lr0.005_error0.1_diag1_sos0.0_state0.5_0.5_0.0_0.0/0/'

# path = '/run/media/trevor/01CA-028A/nips_kalman/5_17_18r0.1/kf1_noisyobj0_fparabola_maxsamples5000_batch5000_lr0.01_error0.1_diag0_sos0.0/5/'
# path = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/kf1_noisyobj0_fparabola_maxsamples5000_batch5000_lr0.01_error0.1_diag1_sos0.0/0/'
# path = '/home/trevor/Documents/dev/ml/rl/rl-adaptive-sampling/rl_adaptive_sampling/bandit/data/batch1000_lr0.1_error0.025_noisyobj0_fparabola_diag1_sos0.0/1/'

obs_noise = np.load(os.path.join(path, 'log_est_obs_noise.npy'))
# print (obs_noise.shape)
obs_noise = np.sum(obs_noise, axis=1)
# print (obs_noise.shape)
# input("")
# obs_noise = np.load(os.path.join(path, 'log_obs_noise_est.npy'))
batches = np.load(os.path.join(path, 'log_batch_sizes.npy'))
batch_ends = np.concatenate((np.array([0]), np.cumsum(batches)))[0:50]

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

plt.xlim((0, 50))
plt.legend()
ax1.legend(loc=2)
fig.tight_layout()
# plt.show()
plt.savefig(fname='batch_var_lqr.pdf', format='pdf')
