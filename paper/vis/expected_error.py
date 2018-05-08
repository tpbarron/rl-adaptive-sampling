import os

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(4, 3))

path = '/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_7_18r1/'
path2 = '/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_7_18r1/'

# func = 'parabola'
func = 'ndquad'
use_diagonal_approx = 1
seeds = list(range(5))
lr = 0.1

# kalman
# errs = [0.2, 0.1, 0.05, 0.01]
lrs = [0.1]
errs = [0.2]#, 0.1]
colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
sos_init = 25.0

for e, c in zip(errs, colors):
    xs = []
    ys = []
    for seed in [0]: #, 1, 2, 3, 4]:
        batch_sizes1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_batch_sizes.npy')

        errors = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_abs_error_est.npy')

        batch_ends = np.concatenate((np.array([0]), np.cumsum(batch_sizes1)))
        for i in range(5): #len(batch_ends)-1):
            x = np.arange(batch_sizes1[i])[i:]
            # print (x.shape)
            # print (batch_ends[i], batch_ends[i+1])
            # if i == len(batch_ends)-1:
                # end = -1
            # else:
            end = batch_ends[i+1]
            y = errors[batch_ends[i]:end]
            y = np.mean(y[:,:,0], axis=1)[i:]
            # print (y.shape)
            # input("")
            plt.plot(x, y, color='black', label='error estimate')
    # convert xs, ys to same length

plt.xlabel("Iteration")
plt.ylabel("Error estimate")
# plt.xlim((0, 100))

# plt.legend(prop={'size': 8})

plt.tight_layout()
# plt.show()
plt.savefig(fname='vpg_'+func+'_expected_error.pdf', format='pdf')
