import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import envs.lqg_env as lqg
import envs.lqr_env as lqr
import envs.cartpole as cp
import envs.linearized_cartpole as lcp
import envs.dubins_car as dubins_car
import envs.acrobot_continuous as acrobot_continuous

from rl_adaptive_sampling.rl.policies.linear_policy import LinearPolicy
from rl_adaptive_sampling.rl.baselines.zero_baseline import ZeroBaseline
from rl_adaptive_sampling.rl.baselines.linear_baseline import LinearBaselineParameterized, LinearBaselineKernelRegression, LinearPolynomialKernelBaseline
from rl_adaptive_sampling.rl.filters.kalman_opt import KalmanFilter
from rl_adaptive_sampling.rl import pg
from rl_adaptive_sampling.rl.utils import simple_sampler, eval
from rl_adaptive_sampling.rl.utils.logger import DataLog

def build_log_dir(args):
    args.log_dir = os.path.join(args.log_dir, "kf_"+str(int(not args.no_kalman)))
    if args.no_kalman:
        # append batch size
        args.log_dir = os.path.join(args.log_dir, "bs"+str(args.batch_size_traj))
    else:
        # append error thresh
        args.log_dir = os.path.join(args.log_dir, "kferr"+str(args.kf_error_thresh))
    return args.log_dir

# Start the training process for a variant
def train(args):
    args.log_dir = build_log_dir(args)
    # args.log_dir = os.path.join(args.log_dir, "envlqr"+"_kf"+str(int(not args.no_kalman))+\
    #     "_maxsamples"+str(args.max_samples)+"_batch"+str(args.batch_size)+\
    #     "_lr"+str(args.lr)+"_error"+str(args.kf_error_thresh)+\
    #     "_diag"+str(int(args.use_diagonal_approx))+"_sos"+str(args.sos_init))+\
    #     "_state"+str(args.x0)+"_"+str(args.y0)+"_"+str(args.xv0)+"_"+str(args.yv0)
    args.log_dir = os.path.join(args.log_dir, str(args.seed))
    print ("Starting run with args: ", args)
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log = DataLog()

    # TODO: make env

    # env = lqg.LQG_Env(state0=np.array([args.x0, args.y0, args.xv0, args.yv0]))
    # env = lcp.LinearizedCartPole()
    # K = env.K
    # env = cp.CartPoleContinuousEnv()
    # import gym
    # env = gym.make('Swimmer-v2')
    # env = dubins_car.DubinsCar()
    import gym
    import pybullet_envs
    # import pybullet_envs.bullet.minitaur_gym_env as e
    # env = e.MinitaurBulletEnv(render=False)
    # env = gym.make('MinitaurBulletEnv-v0')
    env = gym.make("Walker2DBulletEnv-v0")
    # import pybullet_envs.bullet.racecarGymEnv as e
    # env = e.RacecarGymEnv(isDiscrete=False, renders=False)
    # env = acrobot_continuous.AcrobotContinuousEnv()
    # env = gym.make('LunarLanderContinuous-v2')

    pi = LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    # pi.lin.weight.data = torch.from_numpy(-K+np.random.normal(scale=3, size=K.shape)).float()
    opt = optim.Adam(pi.parameters(), lr=args.lr)
    # opt = optim.SGD(pi.parameters(), lr=args.lr, momentum=0.9)

    # baseline = LinearPolynomialKernelBaseline(env)
    # baseline = LinearBaselineKernelRegression(env)
    # baseline = LinearBaselineParameterized(env)
    baseline = ZeroBaseline(env)

    kf = KalmanFilter(pi.num_params(),
                      use_diagonal_approx=args.use_diagonal_approx,
                      sos_init=args.sos_init,
                      reset_observation_noise=args.reset_obs_noise,
                      reset_state=args.reset_kf_state,
                      window_size=20)
    kf.reset()

    print ("Random episodes")
    rewards, _, _ = simple_sampler.do_rollout(env, pi, eval=True)
    print ("Episode return: ", sum(rewards))
    rewards, _, _ = simple_sampler.do_rollout(env, pi, optimal=True, eval=True)
    # print ("Optimal return over horizon", env.T,":", sum(rewards))
    print ("Optimal return over horizon:", sum(rewards))

    print ("Test1")
    total_samples = 0
    total_trajs = 0
    tgrads = []
    current_batch_samples = 0
    current_batch_trajs = 0

    eval_perf = eval.eval_pi(env, pi, avg_n=10)
    print ("Initial evaluation avg of 10: ", eval_perf, ", total samples", total_samples)
    # log.log_kv("eval_perf", eval_perf)
    # log.log_kv("total_samples", total_samples)
    # log.log_kv("total_trajs", total_trajs)

    log_batch_sizes = []
    log_est_obs_noise = []
    log_num_trajs = []
    log_num_samples = []
    log_grad_obs = []
    log_eval_perf = [eval_perf]
    batch_grads = []

    i = 0
    while total_samples < args.max_samples:
        # sample data
        rewards, glogps, states = simple_sampler.do_rollout(env, pi)
        tgrads.append([rewards, glogps, states])
        current_batch_samples += len(rewards)
        total_samples += len(rewards)
        current_batch_trajs += 1
        total_trajs += 1

        empirical_state_values = []

        if args.no_kalman:
            # The grad is only computed here to make comparisons.
            opt.zero_grad()
            pg.compute_rollout_grad(args, [(rewards, glogps, states)], baseline, retain=True)
            grad_obs = pi.lin.weight.grad.view(-1).numpy().copy()
            batch_grads.append(grad_obs)
            log_grad_obs.append(grad_obs)

            if current_batch_trajs >= args.batch_size_traj:
            # if current_batch_samples >= args.batch_size:
                opt.zero_grad()
                empirical_state_value = pg.compute_rollout_grad(args, tgrads, baseline)
                empirical_state_values.extend(empirical_state_value)
                opt.step()
                baseline.fit(empirical_state_values)

                # batch_grads = np.array(batch_grads) # grads are (ntraj x nparams)
                # print (batch_grads.shape)
                # import matplotlib.pyplot as plt
                # for i in range(batch_grads.shape[1]):
                #     print (batch_grads[:,i])
                #     plt.hist(batch_grads[:,i], bins=100, density=True)
                #     plt.show()
                batch_grads = []

                eval_perf = eval.eval_pi(env, pi, avg_n=10)
                print ("Evaluation avg of 10: ", eval_perf, ", total samples", total_samples)
                log.log_kv("eval_perf", eval_perf)
                log.log_kv("total_samples", total_samples)
                log.log_kv("total_trajs", total_trajs)

                log_batch_sizes.append(current_batch_trajs)
                log_est_obs_noise.append(np.zeros_like(kf.Rt))
                log_num_samples.append(total_samples)
                log_num_trajs.append(total_trajs)
                log_eval_perf.append(eval_perf)

                tgrads = []
                current_batch_samples = 0
                current_batch_trajs = 0

                # input("")
        else: # not args.no_kalman:
            # check for error thresh
            # print ("KF error: ", np.mean(kf.e))
            opt.zero_grad()
            empirical_state_value = pg.compute_rollout_grad(args, [(rewards, glogps, states)], baseline)
            empirical_state_values.extend(empirical_state_value)
            grad = pi.lin.weight.grad.view(-1, 1).numpy()
            log_grad_obs.append(grad)
            kf.update(grad)
            if np.mean(kf.e) < args.kf_error_thresh:
                print ("KF error: ", np.mean(kf.e), current_batch_trajs)
                pg.execute_kf_grad_step(pi, kf)
                opt.step()
                baseline.fit(empirical_state_values)

                eval_perf = eval.eval_pi(env, pi, avg_n=10)
                print ("Evaluation avg of 10: ", eval_perf, ", total samples", total_samples)
                log.log_kv("eval_perf", eval_perf)
                log.log_kv("total_samples", total_samples)
                log.log_kv("total_trajs", total_trajs)
                log.log_kv("kf_batch", current_batch_trajs)

                log_batch_sizes.append(current_batch_trajs)
                log_est_obs_noise.append(kf.Rt.copy())
                log_num_samples.append(total_samples)
                log_num_trajs.append(total_trajs)
                log_eval_perf.append(eval_perf)

                kf.reset()
                current_batch_samples = 0
                current_batch_trajs = 0
                tgrads = []
        i+=1
        if i % 10 == 0:
            # Log periodically to be safe
            log.save_log(args.log_dir)


    np.save(os.path.join(args.log_dir, "log_batch_sizes.npy"), np.array(log_batch_sizes))
    np.save(os.path.join(args.log_dir, "log_est_obs_noise.npy"), np.array(log_est_obs_noise))
    np.save(os.path.join(args.log_dir, "log_num_trajs.npy"), np.array(log_num_trajs))
    np.save(os.path.join(args.log_dir, "log_num_samples.npy"), np.array(log_num_samples))
    np.save(os.path.join(args.log_dir, "log_grad_obs.npy"), np.stack(log_grad_obs))
    np.save(os.path.join(args.log_dir, "log_eval_perf.npy"), np.array(log_eval_perf))

if __name__ == '__main__':
    import arguments
    args = arguments.get_args()
    train(args)
