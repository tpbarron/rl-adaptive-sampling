import arguments
import os
import time

import ray
from rl_adaptive_sampling.rl import vpg_traj_kf

#BASE_LOG_DIR = "/home/trevor/Documents/data/rl_adaptive_sampling/"
BASE_LOG_DIR = "/home/dockeruser/DockerShare/tpbarron/data/rl_adaptive_sampling/"
# BASE_LOG_DIR = "/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/rl_adaptive_sampling/"
VPG_LOG_DIR = "vpg/5_10_18r1/"

ray.init(num_cpus=16)

@ray.remote
def run_vpg_traj_variant(args):
    vpg_traj_kf.optimize(args)


gets = []
seeds = list(range(10))
envs = ['CartPole-v0']

max_samples = 100000
log_dir = os.path.join(BASE_LOG_DIR, VPG_LOG_DIR)
lrs = [0.05, 0.01]
batch_sizes = [1000, 500, 250, 100]
pi_optim = ['adam'] # 'sgd']

errs = [0.001, 0.0001, 0.00001]
diagonal = [True]
sos_init = [0.001, 0.005, 0.01]
reset_state = [True, False]
reset_obs_noise = [True, False]

print ("No kalman: ", len(seeds) * len(lrs) * len(batch_sizes))
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
                   
                   pid = run_vpg_traj_variant.remote(args)
                   gets.append(pid)

                   
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
                                    
                                    pid = run_vpg_traj_variant.remote(args)
                                    gets.append(pid)

# wait for all processes
ray.get([pid for pid in gets])
