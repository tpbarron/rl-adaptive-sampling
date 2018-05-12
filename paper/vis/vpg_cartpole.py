import os

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(4, 3))

algo = 'npg'

path =  '/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_10_18r1'
# path2 = '/home/trevor/Documents/data/rl_adaptive_sampling/'+algo+'/5_7_18r1/'

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
        # print (ynew.shape)
        # input("")
        xsnew.append(xnew)
        ysnew.append(ynew)
    return xsnew[0], np.array(ysnew)

data_map = {}
title_map = {}
import os
for root, subdirs, files in os.walk(path):
    if "log.csv" in files:
        # print (root, subdirs, files)
        # input("")
        key = root[:-2]
        if not key in data_map:
            data_map[key] = []
            title_map[key] = "KF" if "batch5000" in root else "PG" #key
            title_map[key] += root[root.find("batch"): root.find("_", root.find("batch")+1)]
        data_map[key].append(os.path.join(root, "log.csv"))

# print (data_map)
# input("")

for k,v in data_map.items():
    xs = []
    ys = []
    print (len(v))
    for file in v:
        a = np.loadtxt(file, delimiter=',')
        xs.append(a[:,0])
        ys.append(a[:,3])

    xsnew, ysnew = sync_data(xs, ys)
    print ("Run data: ", xsnew.shape, ysnew.shape)
    ysnew = np.mean(ysnew, axis=0)
    print ("Run data: ", xsnew.shape, ysnew.shape)
    # mean = np.stack(run_data)
    # print (mean.shape)
    # mean = np.mean(mean, axis=1)
    # print (mean.shape)
    plt.plot(xsnew, ysnew, label=title_map[k])

plt.legend()
# plt.show()
plt.savefig(fname='vpg_cartpole.pdf', format='pdf')
input("")

func = 'parabola'
# func = 'ndquad'
use_diagonal_approx = 1
seeds = list(range(5))
lr = 0.05

# no kalman
bs = [1000, 500, 250, 100]
colors = ['#c2e699', '#78c679', '#31a354', '#006837']
for b, c in zip(bs, colors):
    # batch 1000
    xs = []
    ys = []
    for s in seeds:
        batch_sizes1 = np.load(path+'batch'+str(b)+'_lr'+str(lr)+'_error0.0_noisyobj0_f'+func+'_diag0_sos0.0/'+str(s)+'/log_batch_sizes.npy')
        mu_est1 = np.load(path+'batch'+str(b)+'_lr'+str(lr)+'_error0.0_noisyobj0_f'+func+'_diag0_sos0.0/'+str(s)+'/log_min_mu_est.npy')
        batch_ends1 = np.cumsum(batch_sizes1)
        # print (batch_ends1)
        x = batch_ends1
        y = mu_est1[batch_ends1]**2.0
        y = np.mean(y, axis=1)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    x = np.mean(xs, axis=0)
    y = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)
    plt.plot(x, y, label='PG '+str(b), color=c)
    plt.fill_between(x, y, y+std, alpha=0.25, color=c)
    plt.fill_between(x, y, y-std, alpha=0.25, color=c)
    # plt.errorbar(x, y, yerr=std, color=c)

# kalman
errs = [0.2, 0.1, 0.05, 0.01]
lrs = [0.1]
errs = [0.2, 0.1]
colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
sos_init = 10.0


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
        # print (ynew.shape)
        # input("")
        xsnew.append(xnew)
        ysnew.append(ynew)
    return xsnew[0], np.array(ysnew)

for e, c in zip(errs, colors):
    xs = []
    ys = []
    for seed in seeds:
        batch_sizes1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_batch_sizes.npy')
        mu_est1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/'+str(seed)+'/log_min_mu_est.npy')
        batch_ends1 = np.cumsum(batch_sizes1)
        # batch_ends1 = np.concatenate((np.array([0]), batch_ends1))
        x = batch_ends1
        y = mu_est1[batch_ends1]**2.0
        y = np.mean(y, axis=1)
        xs.append(x)
        ys.append(y)
    # convert xs, ys to same length
    xsnew, ysnew = sync_data(xs, ys)
    print (ysnew.shape)
    y = np.mean(ysnew, axis=0)
    ystd = np.std(ysnew, axis=0)
    plt.plot(xsnew, y, label='PG-KF '+str(e), color=c)
    plt.fill_between(xsnew, y, y+ystd, alpha=0.25, color=c)
    plt.fill_between(xsnew, y, y-ystd, alpha=0.25, color=c)

plt.xlabel("Samples")
plt.ylabel("Squared Error")

if func == 'parabola':
    plt.xlim((0, 5000))
elif func == 'ndquad':
    plt.xlim((0, 10000))
plt.ylim((0, 1.0))

plt.legend(prop={'size': 8})
plt.tight_layout()

plt.show()
# plt.savefig(fname=algo+'_'+func+'.pdf', format='pdf')
