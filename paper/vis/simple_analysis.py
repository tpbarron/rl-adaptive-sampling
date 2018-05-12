import os

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(4, 3))

algo = 'brs'
#path = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_6_18r3/'
# path = '/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/vpg/5_7_18r3/'
# path2 = '/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/vpg/5_7_18r3/'
path = '/home/trevor/Documents/data/rl_adaptive_sampling/'+algo+'/5_7_18r1/'
path2 = '/home/trevor/Documents/data/rl_adaptive_sampling/'+algo+'/5_7_18r1/'

func = 'parabola'
# func = 'ndquad'
use_diagonal_approx = 1
noisy_obj = 1
seeds = list(range(1))
lr = 0.05
alpha = 0.2
sos_init = 0.0
nu = 0.5
name = algo+'_'+func+'_sos'+str(sos_init)+'noisy'+str(noisy_obj)+'_1'

# no kalman
# bs = [250, 100] #, 50, 10]
bs = [1, 10, 50]
colors = ['xkcd:orange', 'xkcd:orange red', 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']
markers = [',', ',', ',']
for b, c, m in zip(bs, colors, markers):
    # batch 1000
    xs = []
    ys = []
    for s in seeds:
        if algo == 'brs':
            batch_sizes1 = np.load(path+'batch'+str(b)+'_lr'+str(lr)+'_error0.0_noisyobj'+str(noisy_obj)+'_f'+func+'_diag0_sos0.0_nu'+str(nu)+'/'+str(s)+'/log_batch_sizes.npy')
            mu_est1 = np.load(path+'batch'+str(b)+'_lr'+str(lr)+'_error0.0_noisyobj'+str(noisy_obj)+'_f'+func+'_diag0_sos0.0_nu'+str(nu)+'/'+str(s)+'/log_min_est.npy')
        else:
            batch_sizes1 = np.load(path+'batch'+str(b)+'_lr'+str(lr)+'_error0.0_noisyobj'+str(noisy_obj)+'_f'+func+'_diag0_sos0.0/'+str(s)+'/log_batch_sizes.npy')
            mu_est1 = np.load(path+'batch'+str(b)+'_lr'+str(lr)+'_error0.0_noisyobj'+str(noisy_obj)+'_f'+func+'_diag0_sos0.0/'+str(s)+'/log_min_mu_est.npy')
        batch_ends1 = np.cumsum(batch_sizes1)
        # print (batch_ends1)
        x = batch_ends1
        y = mu_est1[batch_ends1]**2.0
        # y = np.reshape(1, y.shape[0])
        # print (y.shape)
        y = np.squeeze(y)
        # y = np.mean(y, axis=1)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    x = np.mean(xs, axis=0)
    y = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)**2.0
    plt.plot(x, y, label='PG '+str(b), color=c, marker=m, linestyle='dashed')
    plt.fill_between(x, y, y+std, alpha=alpha, color=c)
    plt.fill_between(x, y, y-std, alpha=alpha, color=c)
    # plt.errorbar(x, y, yerr=std, color=c)

# kalman
# errs = [0.2, 0.1, 0.05, 0.01]
# lr = 0.5
# errs = [0.1, 0.2] #, 0.1]
errs = [0.1, 0.2, 0.3]
colors = ['xkcd:blue', 'xkcd:blue purple', 'xkcd:purple blue'] #, 'xkcd:cobalt blue'] #, '#7fbf7b', '#1b7837']
markers = [',', ',', ','] #, ',']

def sync_data(xs, ys):
    max_samples = 0
    for x in xs:
        max_samples = max(max_samples, np.max(x))
    print ("Max samples", max_samples)

    xsnew = []
    ysnew = []

    for i in range(len(xs)):
        xnew = np.linspace(0, max_samples, max_samples)
        ynew = np.interp(xnew, xs[i], ys[i])
        xsnew.append(xnew)
        ysnew.append(ynew)
    return xsnew[0], np.array(ysnew)

for e, c, m in zip(errs, colors, markers):
    xs = []
    ys = []
    for seed in seeds:
        if algo == 'brs':
            batch_sizes1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj'+str(noisy_obj)+'_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'_nu'+str(nu)+'/'+str(seed)+'/log_batch_sizes.npy')
            mu_est1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj'+str(noisy_obj)+'_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'_nu'+str(nu)+'/'+str(seed)+'/log_min_est.npy')
            batch_ends1 = np.cumsum(batch_sizes1)
        else:
            batch_sizes1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj'+str(noisy_obj)+'_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_batch_sizes.npy')
            mu_est1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj'+str(noisy_obj)+'_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_min_mu_est.npy')
        batch_ends1 = np.cumsum(batch_sizes1)
        # batch_ends1 = np.concatenate((np.array([0]), batch_ends1))
        x = batch_ends1-1
        print (batch_sizes1)
        print (batch_ends1)
        y = mu_est1[x]**2.0
        # y = np.mean(y, axis=1)
        y = np.squeeze(y)
        xs.append(x)
        ys.append(y)
    # convert xs, ys to same length
    xsnew, ysnew = sync_data(xs, ys)
    print (ysnew.shape)
    y = np.mean(ysnew, axis=0)
    ystd = np.std(ysnew, axis=0)**2.0

    # from scipy.interpolate import spline
    # xnew = np.linspace(xsnew.min(), xsnew.max(), 10)
    # ysmooth = spline(xsnew, y, xnew)
    # plt.plot(xnew, ysmooth, label='PG-KF '+str(e), color=c, linestyle='dashed', marker=m)
    plt.plot(xsnew, y, label='PG-KF '+str(e), color=c, marker=m)
    plt.fill_between(xsnew, y, y+ystd, alpha=alpha, color=c)
    plt.fill_between(xsnew, y, y-ystd, alpha=alpha, color=c)

plt.xlabel("Samples")
plt.ylabel("Squared Error")

if func == 'parabola':
    # plt.xlim((0, 3000))
    plt.xlim((0, 1000))
elif func == 'ndquad':
    plt.xlim((0, 10000))
plt.ylim((0, 1.0))

plt.legend(prop={'size': 8})
plt.tight_layout()

plt.show()
# plt.savefig(fname=name+'.pdf', format='pdf')
