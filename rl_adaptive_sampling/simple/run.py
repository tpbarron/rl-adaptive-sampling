import arguments
import os
import time

import ray

import vpg
import brs

RUN_VPG = True
RUN_BRS = False

# BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/"

VPG_LOG_DIR = "vpg/5_17_18r0.1/"
BRS_LOG_DIR = "brs/5_13_18r1/"

ray.init() #num_cpus=3)

@ray.remote
def run_vpg_variant(args):
    vpg.optimize(args)

@ray.remote
def run_brs_variant(args):
    brs.optimize(args)

gets = []
seeds = list(range(10))

if RUN_VPG:
    log_dir = os.path.join(BASE_LOG_DIR, VPG_LOG_DIR)
    lrs = [0.01] #, 0.05]
    batch_sizes = [1, 2, 10, 50, 100, 250, 500, 1000]
    funcs = ['parabola']#, 'ndquad']

    errs = [0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
    noisy_obj = [False]#, True] #, True]
    diagonal = [True, False]
    sos_init = [0.0] #, 0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 100.0]

    print ("Running variants: ", len(seeds) * len(lrs) * len(noisy_obj) * len(funcs) * len(batch_sizes))
    print ("Running variants: ", len(seeds) * len(lrs) * len(errs) * len(noisy_obj) * len(funcs) * len(diagonal) * len(sos_init))
    # no kalman: 240 variants
    for seed in seeds:
       for lr in lrs:
           for nobj in noisy_obj:
               for f in funcs:
                   for bs in batch_sizes:
                       args = arguments.get_args()
                       args.max_samples = 5000
                       args.batch_size = bs
                       args.seed = seed
                       args.lr = lr
                       args.kf_error_thresh = 0.0
                       args.log_dir = log_dir
                       args.no_kalman = True
                       args.noisy_objective = nobj
                       args.func = f
                       pid = run_vpg_variant.remote(args)
                       gets.append(pid)

    # with kalman: 960 variants
    for seed in seeds:
        for lr in lrs:
            for err in errs:
                for nobj in noisy_obj:
                    for diag in diagonal:
                        for f in funcs:
                            for sos in sos_init:
                                args = arguments.get_args()
                                args.max_samples = 5000
                                args.batch_size = 5000
                                args.seed = seed
                                args.lr = lr
                                args.kf_error_thresh = err
                                args.log_dir = log_dir
                                args.no_kalman = False
                                args.use_diagonal_approx = diag
                                args.noisy_objective = nobj
                                args.func = f
                                args.sos_init = sos
                                pid = run_vpg_variant.remote(args)
                                gets.append(pid)

    ray.get([pid for pid in gets])


if RUN_BRS:
    log_dir = os.path.join(BASE_LOG_DIR, BRS_LOG_DIR)
    lrs = [0.01] #, 0.05]
    batch_sizes = [1, 2, 10, 50, 100] #, 250, 500] #, 1000]
    nus = [0.5] #, 0.25, 0.1]
    funcs = ['2dquad']#, 'ndquad']
    noisy_obj = [True, False]#, True]

    errs = [0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
    diagonal = [True]
    sos_init = [0.0] #, 0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 100.0]

    # no kalman
    for seed in seeds:
       for lr in lrs:
           for nobj in noisy_obj:
               for f in funcs:
                   for bs in batch_sizes:
                        for nu in nus:
                            args = arguments.get_args()
                            args.max_samples = 5000
                            args.batch_size = bs
                            args.seed = seed
                            args.lr = lr
                            args.nu = nu
                            args.kf_error_thresh = 0.0
                            args.log_dir = log_dir
                            args.no_kalman = True
                            args.noisy_objective = nobj
                            args.func = f
                            pid = run_brs_variant.remote(args)
                            gets.append(pid)

    # with kalman
    for seed in seeds:
        for lr in lrs:
            for err in errs:
                for nu in nus:
                    for nobj in noisy_obj:
                        for diag in diagonal:
                            for f in funcs:
                                for sos in sos_init:
                                    args = arguments.get_args()
                                    args.max_samples = 5000
                                    args.batch_size = 1000
                                    args.seed = seed
                                    args.lr = lr
                                    args.nu = nu
                                    args.kf_error_thresh = err
                                    args.log_dir = log_dir
                                    args.no_kalman = False
                                    args.use_diagonal_approx = diag
                                    args.noisy_objective = nobj
                                    args.func = f
                                    args.sos_init = sos
                                    pid = run_brs_variant.remote(args)
                                    gets.append(pid)

# wait for all processes
ray.get([pid for pid in gets])
