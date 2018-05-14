import arguments
import os
import time

import ray
from rl_adaptive_sampling.rl import vpg_traj_kf

BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/"
VPG_LOG_DIR_NOKF = "vpg/5_13_18r10_nokf/"
VPG_LOG_DIR_KF = "vpg/5_13_18r10_kf/"

ray.init(num_cpus=3)

@ray.remote
def run_vpg_traj_variant(args):
    vpg_traj_kf.optimize(args)


gets = []
seeds = list(range(10))  #, 5))
envs = ['CartPole-v0']

max_samples = 100000
log_dir = os.path.join(BASE_LOG_DIR, VPG_LOG_DIR_NOKF)
lrs = [0.005]
batch_sizes = [1000, 2500, 5000] #, 1000, 2500, 5000]
pi_optim = ['adam'] # 'sgd']

# errs = [0.0000005] #, 0.0005, 0.0001]
errs = [0.00001, 0.000001, 0.0000005] #, 0.0005, 0.0001]
diagonal = [True]
sos_init = [0.0]
reset_state = [False] #, True]
reset_obs_noise = [False] #, True]

nlayers = 2

# python vpg_traj_kf.py --batch-size 5000 --env-name CartPole-v0 --kf-error-thresh 0.0005 --use-diagonal-approx --sos-init 0.005 --lr 0.01 --seed 1 --pi-optim adam --max-samples 50000 --layers 2

# python vpg_traj_kf.py --batch-size 500 --lr 0.05 --max-samples 100000 --env-name CartPole-v0 --pi-optim adam --layers 1 --no-kalman --seed 0

# --
#  python vpg_traj_kf.py --batch-size 5000 --lr 0.01 --max-samples 100000 --env-name CartPole-v0 --pi-optim adam --layers 2 --seed 0 --sos-init 0.0 --kf-error-thresh 0.001 --reset-kf-state

# print ("No kalman: ", len(seeds) * len(lrs) * len(batch_sizes))
for seed in seeds:
   for lr in lrs:
       for bs in batch_sizes:
           for piopt in pi_optim:
               for e in envs:
                   args = arguments.get_args()
                   args.seed = seed
                   args.env_name = e
                   args.max_samples = max_samples
                   args.batch_size = bs
                   args.lr = lr
                   args.log_dir = log_dir
                   args.pi_optim = piopt

                   # These are just defaults for no kalman
                   args.no_kalman = True
                   args.kf_error_thresh = 0.0
                   args.use_diagonal_approx = False
                   args.sos_init = 0.0
                   args.reset_kf_state = False
                   args.reset_obs_noise = False

                   args.layers = nlayers

                   pid = run_vpg_traj_variant.remote(args)
                   gets.append(pid)

log_dir = os.path.join(BASE_LOG_DIR, VPG_LOG_DIR_KF)

print ("Kalman: ", len(seeds) * len(lrs) * len(errs) * len(sos_init) * len(reset_state) * len(reset_obs_noise))
for seed in seeds:
    for lr in lrs:
        for err in errs:
            for diag in diagonal:
                for sos in sos_init:
                    for piopt in pi_optim:
                        for resetx in reset_state:
                            for reset_obs in reset_obs_noise:
                                for e in envs:
                                    args = arguments.get_args()
                                    args.seed = seed
                                    args.env_name = e
                                    args.max_samples = max_samples
                                    args.batch_size = 5000 # just a large value usually not limiting
                                    args.lr = lr
                                    args.log_dir = log_dir
                                    args.pi_optim = piopt

                                    args.no_kalman = False
                                    args.kf_error_thresh = err
                                    args.use_diagonal_approx = diag
                                    args.sos_init = sos
                                    args.reset_kf_state = resetx
                                    args.reset_obs_noise = reset_obs
                                    args.layers = nlayers

                                    pid = run_vpg_traj_variant.remote(args)
                                    gets.append(pid)

# wait for all processes
ray.get([pid for pid in gets])
