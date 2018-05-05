import arguments
import time
import ray
import vpg
from importlib import reload

ray.init(num_cpus=4)

@ray.remote
def run_variant(args):
    reload(vpg)
    vpg.optimize(args)


log_dir = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/vpg/5_6_18/"
seeds = list(range(10))
lrs = [0.5, 0.3, 0.2, 0.1, 0.05]
errs = [0.5, 0.2, 0.1]
# kf_flag = [False, True]

gets = []

# no kalman
for seed in seeds:
    for lr in lrs:
        args = arguments.get_args()
        args.seed = seed
        args.lr = lr
        args.kf_error_thresh = 0.0
        args.log_dir = log_dir
        args.no_kalman = True
        pid = run_variant.remote(args)
        gets.append(pid)
        # ray.get(run_variant.remote(args))

# with kalman
for seed in seeds:
    for lr in lrs:
        for err in errs:
            args = arguments.get_args()
            args.seed = seed
            args.lr = lr
            args.kf_error_thresh = err
            args.log_dir = log_dir
            args.no_kalman = False
            pid = run_variant.remote(args)
            gets.append(pid)
            # ray.get(run_variant.remote(args))

ray.get([pid for pid in gets])
