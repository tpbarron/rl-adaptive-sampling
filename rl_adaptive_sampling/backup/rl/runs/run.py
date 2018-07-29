import arguments
import os
import time

import ray
import npg
import importlib

# BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
#BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/"
NPG_LOG_DIR = "npg/5_8_18r2/"

ray.init(num_cpus=16)

@ray.remote
def run_npg_variant(args):
    npg.optimize(args)

gets = []
seeds = list(range(3))

log_dir = os.path.join(BASE_LOG_DIR, NPG_LOG_DIR)
lrs = [0.01, 0.05]
errs = [0.001, 0.0005]
diagonal = [True]
batch_sizes = [1000, 500, 250, 100]
sos_init = [100.0, 250.0]
envs = ['Walker2DBulletEnv-v0']

print ("No kalman variants: ", len(seeds) * len(batch_sizes) * len(lrs))
for seed in seeds:
   for lr in lrs:
       for bs in batch_sizes:
           for e in envs:
               args = arguments.get_args()
               args.env_name = e
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

print ("Kalman variants: ", len(seeds) * len(errs) * len(lrs) * len(diagonal) * len(sos_init))
for seed in seeds:
    for lr in lrs:
        for err in errs:
            for diag in diagonal:
                for sos in sos_init:
                    for e in envs:
                        args = arguments.get_args()
                        args.env_name = e
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
