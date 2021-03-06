import arguments
import os
import time

import ray
import train

# BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/aaai/"
BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/aaai/"
# BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/test/"
LQR_LOG_DIR = "walker2d/9_2_18err0.1_lr0.01_polynomial_baseline_gae/"

ray.init()

@ray.remote
def run_lqr_variant(args):
    train.train(args)

gets = []
seeds = list(range(10, 60))

log_dir = os.path.join(BASE_LOG_DIR, LQR_LOG_DIR)
print ("BASE_LOG_DIR: ", log_dir)

diagonal = [True] #, False]
errs = [0.1, 0.2, 0.3]
# batch_sizes = [100, 500, 1000, 5000]
batch_sizes_traj = [1, 2, 5, 10]
# max_samples = 30000
max_samples = 1000000
lr = 0.01
# positions = [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.1, -0.1], [0.5, 0.5, -0.25, 0.5]]
# "--x0 0.5 --y0 0.5"
# "--x0 0.5 --y0 0.5 --xv0 0.1 --yv0 -0.1"

# print ("No kalman variants: ", len(seeds) * len(batch_sizes) * len(positions))
# for seed in seeds:
#    for bs in batch_sizes:
#        for pos in positions:
#            args = arguments.get_args()
#            args.max_samples = 500000
#            args.batch_size = bs
#            args.seed = seed
#            args.lr = 0.005
#            args.x0 = pos[0]
#            args.y0 = pos[1]
#            args.xv0 = pos[2]
#            args.yv0 = pos[3]
#            args.kf_error_thresh = 0.0
#            args.log_dir = log_dir
#            args.no_kalman = True
#            pid = run_lqr_variant.remote(args)
#            gets.append(pid)
#
# print ("Kalman variants: ", len(seeds) * len(errs) * len(diagonal) * len(positions))
# for seed in seeds:
#     for err in errs:
#         for diag in diagonal:
#             for pos in positions:
#                 args = arguments.get_args()
#                 args.max_samples = 500000
#                 args.batch_size = 10000
#                 args.seed = seed
#                 args.lr = 0.005
#                 args.kf_error_thresh = err
#                 args.log_dir = log_dir
#                 args.no_kalman = False
#                 args.use_diagonal_approx = diag
#                 args.x0 = pos[0]
#                 args.y0 = pos[1]
#                 args.xv0 = pos[2]
#                 args.yv0 = pos[3]
#                 pid = run_lqr_variant.remote(args)
#                 gets.append(pid)


print ("No kalman variants: ", len(seeds) * len(batch_sizes_traj))
for seed in seeds:
   for bs in batch_sizes_traj:
       args = arguments.get_args()
       args.max_samples = max_samples
       args.batch_size_traj = bs
       args.seed = seed
       args.lr = lr
       args.kf_error_thresh = 0.0
       args.log_dir = log_dir
       args.no_kalman = True
       pid = run_lqr_variant.remote(args)
       gets.append(pid)

print ("Kalman variants: ", len(seeds) * len(errs) * len(diagonal))
for seed in seeds:
    for err in errs:
        for diag in diagonal:
            args = arguments.get_args()
            args.max_samples = max_samples
            args.batch_size = 10000
            args.seed = seed
            args.lr = lr
            args.kf_error_thresh = err
            args.log_dir = log_dir
            args.no_kalman = False
            args.use_diagonal_approx = diag
            pid = run_lqr_variant.remote(args)
            gets.append(pid)

ray.get([pid for pid in gets])
