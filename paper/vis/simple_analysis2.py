import os

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(4, 3))

# def load_data(root='/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/', tag='batch10000lr0.2error0.01'):
#     """
#     kw args are example
#     """
#     data = []
#     dir = os.path.join(root, tag)
#     # list folders in dir
#     # this corresponds to seeds
#     for f in os.listdir(dir):
#         seed_dir = os.path.join(dir, f)
#         dat = np.load(os.path.join(seed_dir, "log_min_mu_est.npy"))
#         batch = np.load(os.path.join(seed_dir, "log_batch_sizes.npy"))
#         data.append((batch, dat))
#     return data
#
#
# def average_data(data):
#     """
#     data should be list of (batch sizes, statistic) tuples
#     """
#     # convert data to all same length
#     x_pts = np.arange(100)
#     y_pts = np.zeros((100,))
#
#     print (len(data))
#     # data = [data[1]]
#
#     for i in range(100):
#         # get mean of all elements at iter i
#         samples = []
#         for j in range(len(data)):
#             ind = np.cumsum(data[j][0])[i]
#             yind = data[j][1][ind]
#             samples.append(yind)
#             print (yind)
#         ymean = sum(samples) / len(data)
#         # print (ymean)
#         # input("")
#         y_pts[i] = ymean
#     return x_pts, y_pts
#
#
# data1 = load_data()
# data_mean1 = average_data(data1)
# plt.plot(*data_mean1)
#
# data2 = load_data(tag='batch10000lr0.2error0.0')
# data_mean2 = average_data(data2)
# plt.plot(*data_mean2)
#
# plt.show()

# batch_sizes0 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.1/0/log_batch_sizes.npy")
# mu_est0 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.1/0/log_min_mu_est.npy")
# batch_ends0 = np.cumsum(batch_sizes0)
#


#path = '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_6_18r3/'
path = '/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/vpg/5_7_18r4_all/'
path2 = '/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/vpg/temp/'

#func = 'parabola'
func = 'ndquad'
use_diagonal_approx = 1
seeds = list(range(2))
lr = 0.1

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
        print (batch_ends1)
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
    plt.plot(x, y, label='VPG '+str(b), color=c)
    plt.errorbar(x, y, yerr=std, color=c)

# kalman
errs = [0.2, 0.1, 0.05, 0.01]
lrs = [0.1]
errs = [0.2, 0.1]
colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
sos_init = 0.0

for e, c in zip(errs, colors):
    batch_sizes1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/0/log_batch_sizes.npy')
    mu_est1 = np.load(path2+'batch1000_lr'+str(lr)+'_error'+str(e)+'_noisyobj0_f'+func+'_diag'+str(use_diagonal_approx)+'_sos'+str(sos_init)+'/0/log_min_mu_est.npy')
    batch_ends1 = np.cumsum(batch_sizes1)
    print (batch_ends1)
    x = batch_ends1
    y = mu_est1[batch_ends1]**2.0
    y = np.mean(y, axis=1)
    plt.plot(x, y, label='VPG-KF '+str(e), color=c)

plt.xlabel("Samples")
plt.ylabel("Squared Error")

if func == 'parabola':
    plt.xlim((0, 5000))
elif func == 'ndquad':
    plt.xlim((0, 10000))
plt.ylim((0, 1.0))

plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig(fname='vpg_'+func+'.pdf', format='pdf')
