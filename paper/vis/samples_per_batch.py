import os

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(4, 3))

algo = 'npg'

path = '/home/trevor/Documents/data/rl_adaptive_sampling/'+algo+'/5_7_18r1/'
path2 = '/home/trevor/Documents/data/rl_adaptive_sampling/'+algo+'/5_7_18r1/'

func = 'parabola'
# func = 'ndquad'
use_diagonal_approx = 1
seeds = list(range(5))
lr = 0.05

# kalman
# errs = [0.2, 0.1, 0.05, 0.01]
lrs = [0.1]
errs = [0.2]#, 0.1]
colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
sos_init = 100.0

for e, c in zip(errs, colors):
    xs = []
    ys = []
    for seed in [0]: #, 1, 2, 3, 4]:
        batch_sizes1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_batch_sizes.npy')

        variances = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_obs_noise_est.npy')

        batch_variances = variances[np.cumsum(batch_sizes1)][:,:,0]
        print (batch_variances.shape, np.mean(batch_variances, axis=1).shape)
        mean_batch_variances = np.mean(batch_variances, axis=1)
        # print (batch_sizes1)
        # print (variances.shape)
        x = np.arange(len(batch_sizes1))
        plt.plot(x, mean_batch_variances, color='black', label='variance estimate')
        plt.bar(x, batch_sizes1, 1.0, alpha=1.0, color=c) #, label='batch size')
    # convert xs, ys to same length

plt.xlabel("Iteration")
plt.ylabel("Batch Size")
plt.xlim((0, 100))

# if func == 'parabola':
#     plt.xlim((0, 5000))
# elif func == 'ndquad':
#     plt.xlim((0, 10000))
# plt.ylim((0, 1.0))
plt.legend(prop={'size': 8})

plt.tight_layout()
plt.show()
# plt.savefig(fname='vpg_'+func+'.pdf', format='pdf')
