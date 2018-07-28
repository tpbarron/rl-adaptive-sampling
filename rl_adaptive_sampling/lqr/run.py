import arguments
import os
import time

import ray
import pth_pg

# BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/test/"
LQR_LOG_DIR = "lqr/7_27_18r0.1/"

ray.init(num_cpus=2)

@ray.remote
def run_lqr_variant(args):
    pth_pg.optimize(args)

gets = []
seeds = list(range(10))

log_dir = os.path.join(BASE_LOG_DIR, LQR_LOG_DIR)

diagonal = [True, False]
errs = [0.1, 0.2, 0.3, 0.4, 0.5]
batch_sizes = [100, 500, 1000, 5000]
positions = [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.1, -0.1]]
# "--x0 0.5 --y0 0.5"
# "--x0 0.5 --y0 0.5 --xv0 0.1 --yv0 -0.1"

print ("No kalman variants: ", len(seeds) * len(batch_sizes) * len(positions))
for seed in seeds:
   for bs in batch_sizes:
       for pos in positions:
           args = arguments.get_args()
           args.max_samples = 500000
           args.batch_size = bs
           args.seed = seed
           args.lr = 0.005
           args.x0 = pos[0]
           args.y0 = pos[1]
           args.xv0 = pos[2]
           args.yv0 = pos[3]
           args.kf_error_thresh = 0.0
           args.log_dir = log_dir
           args.no_kalman = True
           pid = run_lqr_variant.remote(args)
           gets.append(pid)

print ("Kalman variants: ", len(seeds) * len(errs) * len(diagonal) * len(positions))
for seed in seeds:
    for err in errs:
        for diag in diagonal:
            for pos in positions:
                args = arguments.get_args()
                args.max_samples = 500000
                args.batch_size = 10000
                args.seed = seed
                args.lr = 0.005
                args.kf_error_thresh = err
                args.log_dir = log_dir
                args.no_kalman = False
                args.use_diagonal_approx = diag
                args.x0 = pos[0]
                args.y0 = pos[1]
                args.xv0 = pos[2]
                args.yv0 = pos[3]
                pid = run_lqr_variant.remote(args)
                gets.append(pid)

ray.get([pid for pid in gets])
