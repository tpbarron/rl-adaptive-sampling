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

path1 = '/run/media/trevor/01CA-028A/nips_kalman/lqr/5_17_18r0.1/'
path2 = '/run/media/trevor/01CA-028A/nips_kalman/lqr/5_17_18r0.1/'

# path1 = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/'
# path2 = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_17_18r0.1/'


use_diagonal_approx = 1
seeds = list(range(10))
maxsamples = 500000
lr = 0.005
state = "state0.5_0.5_0.1_-0.1"
alpha = 0.2
sos_init = 0.0

name = 'lqrtest'

# no kalman
bs = [100, 500, 1000] #, 5000]

colors = ['xkcd:coral', 'xkcd:tangerine', 'xkcd:scarlet'] #, 'xkcd:red orange'] #, '#7fbf7b', '#1b7837']
markers = [',', ',', ',']
for b, c, m in zip(bs, colors, markers):
    print ("Batch; ", b)
    # batch 1000
    xs = []
    ys = []
    for s in seeds:
        bpath = os.path.join(path1, 'envlqr_kf0_maxsamples'+str(maxsamples)+'_batch'+str(b)+'_lr'+str(lr)+'_error0.0_diag0_sos0.0_'+state)
        bpath = os.path.join(bpath, str(s))
        batch_sizes1 = np.load(os.path.join(bpath, 'log_batch_sizes.npy'))
        eval_perf = np.load(os.path.join(bpath, 'log_eval_perf.npy'))
        num_samples = np.load(os.path.join(bpath, 'log_num_samples.npy'))

        batch_ends1 = np.concatenate((np.array([0]), num_samples)) / 100
        x = batch_ends1
        y = eval_perf #np.mean(y, axis=1)
        # print (y.shape)
        # input("")
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    ysmin = np.min(ys, axis=1)
    k = 5
    print (np.min(ysmin))
    idx = np.argpartition(ysmin, k)
    # print(idx)
    # print (min_inds)
    print (np.mean(ysmin[idx[:k]]))

    ys = ys[idx[:5]]
    # print (ys.shape)
    # input("")
    x = np.mean(xs, axis=0)
    y = np.mean(ys, axis=0)
    # print ("mean: ", y.shape)
    std = np.std(ys, axis=0)
    plt.plot(x, y, label='PG '+str(b/100), color=c, marker=m, linestyle='dashed')
    plt.fill_between(x, np.maximum(np.zeros_like(y), y-std), y+std, alpha=alpha, color=c, linewidth=1.0)
    # plt.errorbar(x, y, yerr=std, color=c)

# kalman
# errs = [0.4, 0.5]
errs = [0.1, 0.2, 0.3]
colors = ['xkcd:jade', 'xkcd:aqua', 'xkcd:sea blue'] #, 'xkcd:cobalt blue'] #, '#7fbf7b', '#1b7837']
markers = [',', ',', ','] #, ',']

def sync_data(xs, ys):
    max_samples = 0
    for x in xs:
        max_samples = max(max_samples, np.max(x))
    # print ("Max samples", max_samples)

    xsnew = []
    ysnew = []

    for i in range(len(xs)):
        xnew = np.linspace(0, max_samples, max_samples)
        ynew = np.interp(xnew, xs[i], ys[i])
        xsnew.append(xnew)
        ysnew.append(ynew)
    return xsnew[0], np.array(ysnew)

for e, c, m in zip(errs, colors, markers):
    print ("KF err", e)
    xs = []
    ys = []
    for s in seeds:
        bpath = os.path.join(path2, 'envlqr_kf1_maxsamples'+str(maxsamples)+'_batch10000_lr'+str(lr)+'_error'+str(e)+'_diag'+str(use_diagonal_approx)+'_sos0.0_'+state)
        bpath = os.path.join(bpath, str(s))
        batch_sizes1 = np.load(os.path.join(bpath, 'log_batch_sizes.npy'))
        num_samples = np.load(os.path.join(bpath, 'log_num_samples.npy'))
        eval_perf = np.load(os.path.join(bpath, 'log_eval_perf.npy'))

        batch_ends1 = np.concatenate((np.array([0]), num_samples)) / 100
        y = eval_perf
        x = batch_ends1

        xs.append(x)
        ys.append(y)

    # convert xs, ys to same length
    xsnew, ysnew = sync_data(xs, ys)
    # print (ysnew.shape)
    # input("ysnew")
    k = 5
    ysmin = np.min(ysnew, axis=1)
    print (ysmin)
    print (np.min(ysmin))
    idx = np.argpartition(ysmin, k)
    print (np.mean(ysmin[idx[:k]]))
    ysnew = ysnew[idx[:k]]

    y = np.mean(ysnew, axis=0)
    ystd = np.std(ysnew, axis=0)

    # print (xsnew, ysnew)
    # from scipy.interpolate import spline
    # xnew = np.linspace(xsnew.min(), xsnew.max(), 10)
    # ysmooth = spline(xsnew, y, xnew)
    # plt.plot(xnew, ysmooth, label='PG-KF '+str(e), color=c, linestyle='dashed', marker=m)
    plt.plot(xsnew, y, label='PG-KF '+str(e), color=c, marker=m, linewidth=1.0)
    plt.fill_between(xsnew, np.maximum(np.zeros_like(y), y-ystd), y+ystd, alpha=alpha, color=c)


plt.xlabel("Trajectories")
plt.ylabel("Policy Cost")

plt.xlim((0, 5000))
# plt.xlim((0, 50))


# if func == '2dquad':
#     plt.xlim((0, 3000))
#     plt.xlim((0, 1500))
# elif func == 'ndquad':
#     plt.xlim((0, 10000))
plt.ylim((12, 30.0))

plt.legend(prop={'size': 8})
plt.tight_layout()

plt.show()
# plt.savefig(fname='lqr_perf.pdf', format='pdf')
