import arguments
import os
import time

import ray
import score_fn

BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/test/"
LQR_LOG_DIR = "bandit/5_17_18r7/"

ray.init(num_cpus=3)

@ray.remote
def run_score_fn_variant(args):
    score_fn.optimize(args)

gets = []
seeds = list(range(10))

log_dir = os.path.join(BASE_LOG_DIR, LQR_LOG_DIR)

diagonal = [True]#, False]
errs = [0.1, 0.2]#, 0.3, 0.4, 0.5]
batch_sizes = [2, 10, 100, 500, 1000]

lr = 0.01
window_size = 20
max_samples = 1000

# print ("No kalman variants: ", len(seeds) * len(batch_sizes))
# for seed in seeds:
#    for bs in batch_sizes:
#        args = arguments.get_args()
#        args.max_samples = max_samples
#        args.batch_size = bs
#        args.seed = seed
#        args.lr = lr
#        args.kf_error_thresh = 0.0
#        args.log_dir = log_dir
#        args.no_kalman = True
#        pid = run_score_fn_variant.remote(args)
#        gets.append(pid)

print ("Kalman variants: ", len(seeds) * len(errs) * len(diagonal))
for seed in seeds:
    for err in errs:
        for diag in diagonal:
            args = arguments.get_args()
            args.max_samples = max_samples
            args.batch_size = 5000
            args.seed = seed
            args.lr = lr
            args.kf_error_thresh = err
            args.log_dir = log_dir
            args.no_kalman = False
            args.use_diagonal_approx = diag
            args.window_size = window_size
            pid = run_score_fn_variant.remote(args)
            gets.append(pid)

ray.get([pid for pid in gets])
