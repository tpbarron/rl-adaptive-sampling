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

BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/"
VPG_LOG_DIR = "vpg/5_6_18r3/"
NPG_LOG_DIR = "npg/5_6_18r3/"
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
seeds = list(range(3))

if RUN_VPG:
    log_dir = os.path.join(BASE_LOG_DIR, VPG_LOG_DIR)
    lrs = [0.1, 0.05]
    errs = [0.2, 0.1, 0.05, 0.01]
    noisy_obj = [False, True]
    # funcs = ['parabola']
    funcs = ['ndquad'] #, 'quartic', 'ndquad']
    diagonal = [False, True]
    batch_sizes = [500, 250, 100]
    # no kalman
    # for seed in seeds:
    #     for lr in lrs:
    #         for nobj in noisy_obj:
    #             for f in funcs:
    #                 for bs in batch_sizes:
    #                     args = arguments.get_args()
    #                     args.n_iters = 100
    #                     args.batch_size = bs
    #                     args.seed = seed
    #                     args.lr = lr
    #                     args.kf_error_thresh = 0.0
    #                     args.log_dir = log_dir
    #                     args.no_kalman = True
    #                     args.noisy_objective = nobj
    #                     args.func = f
    #                     pid = run_vpg_variant.remote(args)
    #                     gets.append(pid)

    # with kalman
    for seed in seeds:
        for lr in lrs:
            for err in errs:
                for nobj in noisy_obj:
                    for diag in diagonal:
                        for f in funcs:
                            args = arguments.get_args()
                            args.n_iters = 100
                            args.batch_size = 1000
                            args.seed = seed
                            args.lr = lr
                            args.kf_error_thresh = err
                            args.log_dir = log_dir
                            args.no_kalman = False
                            args.use_diagonal_approx = diag
                            args.noisy_objective = nobj
                            args.func = f
                            pid = run_vpg_variant.remote(args)
                            gets.append(pid)
    ray.get([pid for pid in gets])


if RUN_NPG:
    log_dir = os.path.join(BASE_LOG_DIR, NPG_LOG_DIR)
    lrs = [0.2, 0.1, 0.05]
    errs = [0.1, 0.05, 0.01]
    noisy_obj = [False, True]
    funcs = ['parabola'] #, 'ndquad', 'quartic']
    diagonal = [False, True]

    # no kalman
    for seed in seeds:
        for lr in lrs:
            for nobj in noisy_obj:
                for f in funcs:
                    args = arguments.get_args()
                    args.n_iters = 50
                    args.batch_size = 1000
                    args.seed = seed
                    args.lr = lr
                    args.kf_error_thresh = 0.0
                    args.log_dir = log_dir
                    args.no_kalman = True
                    args.noisy_objective = nobj
                    args.func = f
                    pid = run_npg_variant.remote(args)
                    gets.append(pid)

    # with kalman
    for seed in seeds:
        for lr in lrs:
            for err in errs:
                for nobj in noisy_obj:
                    for diag in diagonal:
                        for f in funcs:
                            args = arguments.get_args()
                            args.n_iters = 50
                            args.batch_size = 1000
                            args.seed = seed
                            args.lr = lr
                            args.kf_error_thresh = err
                            args.log_dir = log_dir
                            args.no_kalman = False
                            args.use_diagonal_approx = diag
                            args.noisy_objective = nobj
                            args.func = f
                            pid = run_npg_variant.remote(args)
                            gets.append(pid)
    ray.get([pid for pid in gets])

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
