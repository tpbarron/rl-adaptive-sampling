import arguments
import os
import time

import ray
import vpg
import npg
import brs
# from importlib import reload

RUN_VPG = True
RUN_NPG = False
RUN_BRS = False

BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
VPG_LOG_DIR = "vpg/5_5_18/"
NPG_LOG_DIR = "npg/5_5_18/"
BRS_LOG_DIR = "brs/5_5_18/"

ray.init(num_cpus=4)

@ray.remote
def run_vpg_variant(args):
    vpg.optimize(args)

@ray.remote
def run_npg_variant(args):
    npg.optimize(args)

@ray.remote
def run_brs_variant(args):
    brs.optimize(args)

gets = []

seeds = list(range(2))

if RUN_VPG:
    log_dir = os.path.join(BASE_LOG_DIR, VPG_LOG_DIR)
    lrs = [0.2] #[0.5, 0.3, 0.2, 0.1, 0.05]
    errs = [0.01, 0.1, 0.05] #[0.5, 0.2, 0.1]
    # no kalman
    for seed in seeds:
        for lr in lrs:
            args = arguments.get_args()
            args.seed = seed
            args.lr = lr
            args.kf_error_thresh = 0.0
            args.log_dir = log_dir
            args.no_kalman = True
            args.noisy_objective = True
            pid = run_vpg_variant.remote(args)
            gets.append(pid)
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
                args.noisy_objective = True
                pid = run_vpg_variant.remote(args)
                gets.append(pid)

    ray.get([pid for pid in gets])


if RUN_NPG:
    log_dir = os.path.join(BASE_LOG_DIR, NPG_LOG_DIR)
    lrs = [0.1] #[0.5, 0.3, 0.2, 0.1, 0.05]
    errs = [0.1] #[0.5, 0.2, 0.1]
    # no kalman
    for seed in seeds:
        for lr in lrs:
            args = arguments.get_args()
            args.seed = seed
            args.lr = lr
            args.kf_error_thresh = 0.0
            args.log_dir = log_dir
            args.no_kalman = True
            pid = run_npg_variant.remote(args)
            gets.append(pid)
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
                pid = run_npg_variant.remote(args)
                gets.append(pid)

if RUN_BRS:
    log_dir = os.path.join(BASE_LOG_DIR, BRS_LOG_DIR)
    lrs = [0.5, 0.3, 0.2, 0.1, 0.05]
    errs = [0.5, 0.2, 0.1]
    nus = [0.5, 0.25, 0.1, 0.05]
    # no kalman
    for seed in seeds:
        for lr in lrs:
            for nu in nus:
                args = arguments.get_args()
                args.seed = seed
                args.lr = lr
                args.kf_error_thresh = 0.0
                args.log_dir = log_dir
                args.nu = nu
                args.no_kalman = True
                pid = run_brs_variant.remote(args)
                gets.append(pid)
    # with kalman
    for seed in seeds:
        for lr in lrs:
            for err in errs:
                for nu in nus:
                    args = arguments.get_args()
                    args.seed = seed
                    args.lr = lr
                    args.kf_error_thresh = err
                    args.log_dir = log_dir
                    args.no_kalman = False
                    args.nu = nu
                    pid = run_brs_variant.remote(args)
                    gets.append(pid)


# wait for all processes
ray.get([pid for pid in gets])
