import arguments
import os
import time

import ray
import npg
import importlib

# BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/"
NPG_LOG_DIR = "npg/5_7_18r1/"

ray.init()

@ray.remote
def run_npg_variant(args):
    npg.optimize(args)

gets = []
seeds = list(range(3))

log_dir = os.path.join(BASE_LOG_DIR, NPG_LOG_DIR)
lrs = [0.1, 0.05]
errs = [0.01, 0.001]
diagonal = [True]
batch_sizes = [1000, 500, 250, 100]
sos_init = [1000.0, 100.0]

for seed in seeds:
   for lr in lrs:
       for bs in batch_sizes:
           args = arguments.get_args()
           args.max_samples = 100000
           args.batch_size = bs
           args.seed = seed
           args.lr = lr
           args.kf_error_thresh = 0.0
           args.log_dir = log_dir
           args.no_kalman = True
           args.sos_init = 0.0
           pid = run_npg_variant.remote(args)
           gets.append(pid)

# with kalman: 960 variants
for seed in seeds:
    for lr in lrs:
        for err in errs:
            for diag in diagonal:
                for sos in sos_init:
                    args = arguments.get_args()
                    args.max_samples = 100000
                    args.batch_size = 1000
                    args.seed = seed
                    args.lr = lr
                    args.kf_error_thresh = err
                    args.log_dir = log_dir
                    args.no_kalman = False
                    args.use_diagonal_approx = diag
                    args.sos_init = sos
                    pid = run_npg_variant.remote(args)
                    gets.append(pid)
ray.get([pid for pid in gets])
