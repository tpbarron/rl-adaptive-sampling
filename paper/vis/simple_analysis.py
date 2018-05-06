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


batch_sizes1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.1noisyobjTrue/1/log_batch_sizes.npy")
mu_est1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.1noisyobjTrue/1/log_min_mu_est.npy")
batch_ends1 = np.cumsum(batch_sizes1)
x = batch_ends1
y = mu_est1[batch_ends1]**2.0
plt.plot(x, y, label='VPG-KF 0.1 noisy')

batch_sizes1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.05noisyobjTrue/1/log_batch_sizes.npy")
mu_est1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.05noisyobjTrue/1/log_min_mu_est.npy")
batch_ends1 = np.cumsum(batch_sizes1)
x = batch_ends1
y = mu_est1[batch_ends1]**2.0
plt.plot(x, y, label='VPG-KF 0.05 noisy')


batch_sizes1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.01noisyobjTrue/1/log_batch_sizes.npy")
mu_est1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.01noisyobjTrue/1/log_min_mu_est.npy")
batch_ends1 = np.cumsum(batch_sizes1)
x = batch_ends1
y = mu_est1[batch_ends1]**2.0
plt.plot(x, y, label='VPG-KF 0.01 noisy')

batch_sizes1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.1/1/log_batch_sizes.npy")
mu_est1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.1/1/log_min_mu_est.npy")
batch_ends1 = np.cumsum(batch_sizes1)
x = batch_ends1
y = mu_est1[batch_ends1]**2.0
plt.plot(x, y, label='VPG-KF 0.1')


batch_sizes1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.05/1/log_batch_sizes.npy")
mu_est1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.05/1/log_min_mu_est.npy")
batch_ends1 = np.cumsum(batch_sizes1)
x = batch_ends1
y = mu_est1[batch_ends1]**2.0
plt.plot(x, y, label='VPG-KF 0.05')

batch_sizes1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.01/1/log_batch_sizes.npy")
mu_est1 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.01/1/log_min_mu_est.npy")
batch_ends1 = np.cumsum(batch_sizes1)
x = batch_ends1
y = mu_est1[batch_ends1]**2.0
plt.plot(x, y, label='VPG-KF 0.01')


batch_sizes0 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.0/0/log_batch_sizes.npy")[0:10]
mu_est0 = np.load("/home/trevor/Documents/data/rl_adaptive_sampling/vpg/5_5_18/batch10000lr0.2error0.0/0/log_min_mu_est.npy")
batch_ends0 = np.cumsum(batch_sizes0)
plt.plot(batch_ends0, mu_est0[batch_ends0]**2.0, label='VPG')

plt.xlabel("Samples")
plt.ylabel("Squared Error")

plt.legend()
plt.tight_layout()

plt.show()
# plt.savefig(fname='simple_quadratic.pdf', format='pdf')
