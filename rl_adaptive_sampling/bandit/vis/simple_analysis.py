import os

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set_style("white", {"legend.frameon": True})

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(4, 3))

# path1 = '/home/trevor/Documents/data/rl_adaptive_sampling/bandit/5_17_18r1/'
# path2 = '/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_17_18r11/'

path1 = '/run/media/trevor/01CA-028A/nips_kalman/5_17_18r0.1/'
path2 = '/run/media/trevor/01CA-028A/nips_kalman/5_17_18r0.1/'

# path1 = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/'
# path2 = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/'


func = 'parabola'
use_diagonal_approx = 1
noisy_obj = 0
seeds = list(range(10))
lr = 0.01
alpha = 0.2
sos_init = 0.0
name = 'f'+func+'_smallbatch'

# no kalman
# bs = [500, 1000] #, 50, 10]
# bs = [1, 2, 10]
# bs = [100, 250, 500, 1000]
bs = [2, 10, 50] #, 100] #, 100]
colors = ['xkcd:coral', 'xkcd:tangerine', 'xkcd:scarlet'] #, 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']
markers = [',', ',', ',']
for b, c, m in zip(bs, colors, markers):
    # batch 1000
    xs = []
    ys = []
    for s in seeds:
        bpath = os.path.join(path1, 'kf0_noisyobj0_fparabola_maxsamples5000_batch'+str(b)+'_lr'+str(lr)+'_error0.0_diag0_sos0.0')
        bpath = os.path.join(bpath, str(s))
        batch_sizes1 = np.load(os.path.join(bpath, 'log_batch_sizes.npy'))
        mu_est1 = np.load(os.path.join(bpath, 'log_min_mu_est.npy'))

        batch_ends1 = np.cumsum(batch_sizes1)
        # print (batch_ends1)
        x = batch_ends1
        y = mu_est1[batch_ends1]**2.0
        # y = np.reshape(1, y.shape[0])
        # y = np.squeeze(y)
        y = np.mean(y, axis=1)
        # print (y.shape)
        # input("")
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    x = np.mean(xs, axis=0)
    y = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)
    plt.plot(x, y, label='PG '+str(b), color=c, marker=m, linestyle='dashed')
    # if bs != 2:
    plt.fill_between(x, y-std, y+std, alpha=alpha, color=c)
    # plt.errorbar(x, y, yerr=std, color=c)

# kalman
errs = [0.4, 0.5, 0.75]
lr = 0.01
# errs = [0.1, 0.2, 0.3]
colors = ['xkcd:jade', 'xkcd:aqua', 'xkcd:sea blue'] #, 'xkcd:cobalt blue'] #, '#7fbf7b', '#1b7837']
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
    for s in seeds:

        bpath = os.path.join(path2, 'kf1_noisyobj0_fparabola_maxsamples5000_batch5000_lr'+str(lr)+'_error'+str(e)+'_diag'+str(use_diagonal_approx))+'_sos0.0'
        bpath = os.path.join(bpath, str(s))
        batch_sizes1 = np.load(os.path.join(bpath, 'log_batch_sizes.npy'))
        mu_est1 = np.load(os.path.join(bpath, 'log_min_mu_est.npy'))

        batch_ends1 = np.cumsum(batch_sizes1)
        # batch_ends1 = np.concatenate((np.array([0]), batch_ends1))
        x = batch_ends1-1
        print (batch_sizes1)
        print (batch_ends1)
        y = mu_est1[x]
        y = np.mean(y, axis=1)**2.0
        # y = np.squeeze(y)
        xs.append(x)
        ys.append(y)
    # convert xs, ys to same length
    xsnew, ysnew = sync_data(xs, ys)
    print (ysnew.shape)
    y = np.mean(ysnew, axis=0)
    ystd = np.std(ysnew, axis=0)

    # from scipy.interpolate import spline
    # xnew = np.linspace(xsnew.min(), xsnew.max(), 10)
    # ysmooth = spline(xsnew, y, xnew)
    # plt.plot(xnew, ysmooth, label='PG-KF '+str(e), color=c, linestyle='dashed', marker=m)
    plt.plot(xsnew, y, label='PG-KF '+str(e), color=c, marker=m)
    plt.fill_between(xsnew, y, y+ystd, alpha=alpha, color=c)
    plt.fill_between(xsnew, y, y-ystd, alpha=alpha, color=c)

plt.xlabel("Samples")
plt.ylabel("Squared Error")

plt.xlim((0, 500))
# plt.xlim((0, 1500))


# if func == '2dquad':
#     plt.xlim((0, 3000))
#     plt.xlim((0, 1500))
# elif func == 'ndquad':
#     plt.xlim((0, 10000))
plt.ylim((0, 1.0))

plt.legend(prop={'size': 8})
plt.tight_layout()

# plt.show()
plt.savefig(fname=name+'.pdf', format='pdf')
